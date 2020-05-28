import tensorflow as tf
import numpy as np
import yaml
import os
import h5py
import time
import sys
import math

from Lib.Data_Processing import *
from Lib.Utility import *
from Model.E2E_CNN_model import Depth_Decoder
from Model.Base_Handler import Basement_Handler


class Decoder_Handler(Basement_Handler):
    def __init__(self, dataset_name, model_config, sess, is_training=True):
        
        # Initialization of Configuration, Parameter and Datasets
        super(Decoder_Handler, self).__init__(sess=sess, model_config=model_config, is_training=is_training)
        self.initial_parameter()
        self.data_assignment(dataset_name)

        # Data Generator
        self.gen_train = Data_Generator_File(dataset_name,self.train_index,self.sense_mask,self.batch_size,self.nF,is_training=True,is_valid=False,is_testing=False)
        self.gen_valid = Data_Generator_File(dataset_name,self.valid_index,self.sense_mask,self.batch_size,self.nF,is_training=False,is_valid=True,is_testing=False)
        self.gen_test = Data_Generator_File(dataset_name,self.test_index,self.sense_mask,self.batch_size,self.nF,is_training=False,is_valid=False,is_testing=True)
        
        # Define the general model and the corresponding input
        shape_meas = (self.batch_size, self.sense_mask.shape[0], self.sense_mask.shape[1],self.nF)
        shape_sense = self.sense_mask.shape
        shape_truth = (self.batch_size,) + self.sense_mask.shape
        self.meas_sample = tf.placeholder(tf.float32, shape=shape_meas, name='input_meas')
        self.sense_matrix = tf.placeholder(tf.float32, shape=shape_sense, name='input_mat')
        self.truth_seg = tf.placeholder(tf.float32, shape=shape_truth, name='output_truth')
        
        # Initialization for the model training procedure.
        self.learning_rate = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(self.lr_init),trainable=False)
        self.lr_new = tf.placeholder(tf.float32, shape=(), name='lr_new')
        self.lr_update = tf.assign(self.learning_rate, self.lr_new, name='lr_update')
        self.train_test_valid_assignment()
        # self.trainable_parameter_info() # print network param, zzh
        self.saver = tf.train.Saver(tf.global_variables())

    def initial_parameter(self):
        # Configuration Set
        config = self.model_config
        
        # Model Input Initialization
        self.nF = int(config.get('compressive_ratio',8))
        self.batch_size = int(config.get('batch_size',1))
        self.upbound = float(config.get('upbound',1))
        
        # Initialization for Training Controler
        self.epochs = int(config.get('epochs',100))
        self.patience = int(config.get('patience',30))
        self.lr_init = float(config.get('learning_rate',0.001))
        self.lr_decay_coe = float(config.get('lr_decay',0.1))
        self.lr_decay_epoch = int(config.get('lr_decay_epoch',20))
        self.lr_decay_interval = int(config.get('lr_decay_interval',10))

    def data_assignment(self,dataset_name):
        # Division for train, test and validation
        model_config = self.model_config
        train_index, valid_index, test_index, self.sense_mask = Data_Division(dataset_name,self.nF)
        
        
        # The value of the position is normalized (the value of lat and lon are all limited in range(0,1))
        self.scalar = limit_scalar(self.upbound,self.sense_mask)
        self.train_index,disp_train = train_index, len(train_index)
        print('training_sample_number is '+ str(disp_train))
        self.valid_index,disp_valid = valid_index, len(valid_index)
        print('valid_sample_number is '+ str(disp_valid))
        self.test_index,disp_test = test_index, len(test_index)
        print('testing_sample_number is '+ str(disp_test))
        
        self.train_size = int(np.ceil(float(disp_train)/self.batch_size))
        self.test_size  = int(np.ceil(float(disp_test)/self.batch_size))
        self.valid_size = int(np.ceil(float(disp_valid)/self.batch_size))
        
        # Display the data structure of Training/Testing/Validation Dataset
        print('Available samples (batch) train %d(%d), valid %d(%d), test %d(%d)' % (
            disp_train,self.train_size,disp_valid,self.valid_size,disp_test,self.test_size))
        
    def train_test_valid_assignment(self):#, is_training = True, reuse = False
        
        value_set = (self.meas_sample,
                     tf.expand_dims(self.sense_matrix,0),
                     self.truth_seg)
        
        with tf.name_scope('Train'):
            with tf.variable_scope('Depth_Decoder', reuse=False):
                self.Decoder_train = Depth_Decoder(value_set,self.learning_rate,self.sess,self.model_config,is_training=True)
        with tf.name_scope('Val'):
            with tf.variable_scope('Depth_Decoder', reuse=True):
                self.Decoder_valid = Depth_Decoder(value_set,self.learning_rate,self.sess,self.model_config,is_training=False)
        with tf.name_scope('Test'):
            with tf.variable_scope('Depth_Decoder', reuse=True):
                self.Decoder_test = Depth_Decoder(value_set,self.learning_rate,self.sess,self.model_config,is_training=False)
                
                
    def train(self):
        self.sess.run(tf.global_variables_initializer())
        print ('Training Started')
        if self.model_config.get('model_filename',None) is not None:
            self.restore()
            print ('Pretrained Model Downloaded')
        else:
            print ('New Model Training')
        epoch_cnt,wait,min_val_loss,max_val_psnr = 0,0,float('inf'),0
        Tloss_list,Vloss_list=[],[]
        
        while epoch_cnt <= self.epochs:
            
            # Training Preparation: Learning rate pre=setting, Model Interface summary.
            start_time = time.time()
            cur_lr = self.calculate_scheduled_lr(epoch_cnt)
            train_fetches = {'global_step': tf.train.get_or_create_global_step(), 
                             'train_op':self.Decoder_train.train_op,
                             'metrics':self.Decoder_train.metrics,
                             'pred_orig':self.Decoder_train.decoded_image,
                             'loss':self.Decoder_train.loss}
            valid_fetches = {'global_step': tf.train.get_or_create_global_step(),
                            'pred_orig':self.Decoder_valid.decoded_image,
                             'metrics':self.Decoder_valid.metrics,
                            'loss':self.Decoder_valid.loss}
            Tresults,Vresults = {"loss":[],"psnr":[],"ssim":[]},{"loss":[],"psnr":[],"ssim":[]}
            
            # Framework and Visualization SetUp for Training 
            for trained_batch in range(0,self.train_size):
                (measure_train,mask_train,ground_train) = self.gen_train.__next__()
                feed_dict_train = {self.meas_sample: measure_train, 
                                   self.sense_matrix: mask_train,
                                   self.truth_seg: ground_train}
                train_output = self.sess.run(train_fetches,feed_dict=feed_dict_train)
                Tresults["loss"].append(train_output['loss'])
                Tresults["psnr"].append(train_output['metrics'][0])
                Tresults["ssim"].append(train_output['metrics'][1])
                if trained_batch%500 == 0 and trained_batch != 0:
                    Train_loss = np.mean(np.asarray(Tresults["loss"][-500:]))
                    Train_psnr = np.mean(np.asarray(Tresults["psnr"][-500:]))
                    message = "Train Epoch [%2d/%2d] Batch [%d/%d] lr: %.4f, loss: %.8f psnr: %.4f" % (epoch_cnt, self.epochs, trained_batch, self.train_size, cur_lr, Train_loss, Train_psnr)
                    print(message)
                
            # Framework and Visualization SetUp for Validation 
            list_truth,list_pred = [],[]
            validation_time = []
            for valided_batch in range(0,self.valid_size):
                # (measure_valid,mask_valid,ground_valid) = self.gen_valid.next()
                (measure_valid,mask_valid,ground_valid) = self.gen_valid.__next__() # zzh
                feed_dict_valid = {self.meas_sample: measure_valid,
                                   self.sense_matrix: mask_valid,
                                   self.truth_seg: ground_valid}
                start_time = time.time()
                valid_output = self.sess.run(valid_fetches,feed_dict=feed_dict_valid)
                end_time = time.time()
                validation_time.append(end_time-start_time)

                Vresults["loss"].append(valid_output['loss'])
                Vresults["psnr"].append(valid_output['metrics'][0])
                Vresults["ssim"].append(valid_output['metrics'][1])
                list_truth.append(ground_valid)
                list_pred.append(valid_output['pred_orig'])

            
            # Information Logging for Model Training and Validation (Maybe for Curve Plotting)
            Tloss,Vloss = np.mean(Tresults["loss"]),np.mean(Vresults["loss"])
            train_psnr = np.mean(np.asarray(Tresults["psnr"]))
            valid_psnr_sample = np.reshape(np.asarray(Vresults["psnr"]),self.valid_size*self.batch_size)
            psnr_aerial = np.mean(valid_psnr_sample[0:4])
            psnr_crash = np.mean(valid_psnr_sample[4:8])
            psnr_drop = np.mean(valid_psnr_sample[8])
            psnr_kobe = np.mean(valid_psnr_sample[9:13])
            psnr_runner = np.mean(valid_psnr_sample[13])
            psnr_traffic = np.mean(valid_psnr_sample[14:20])
            valid_psnr = (psnr_aerial+psnr_crash+psnr_drop+psnr_kobe+psnr_runner+psnr_traffic)/6
            train_ssim,valid_ssim = np.mean(Tresults["ssim"]),np.mean(Vresults["ssim"])
            #train_mse, valid_mse  = np.mean(Tresults["mse"]), np.mean(Vresults["mse"])
            summary_format = ['loss/train_loss','loss/valid_loss','metric/train_psnr','metric/train_ssim',
                              'metric/valid_psnr','metric/valid_ssim']
            summary_data = [Tloss, Vloss, train_psnr, train_ssim, valid_psnr, valid_ssim]

            end_time = time.time()
            message = 'Epoch [%3d/%3d] Train(Valid) loss: %.4f(%.4f), PSNR: %s(%s), time %s' % (
                epoch_cnt, self.epochs, Tloss, Vloss, train_psnr, valid_psnr, np.mean(validation_time))
            self.logger.info(message)
            
            if valid_psnr >= max_val_psnr and valid_psnr>=28:
                matcontent = {}
                matcontent[u'truth'],matcontent[u'pred'] = list_truth, list_pred
                hdf5storage.write(matcontent, '.', self.log_dir+'/Validation_epoch%d.mat'%(epoch_cnt),store_python_metadata=False, matlab_compatible=True)
                max_val_psnr = valid_psnr
            
            Tloss_list.append(Tloss)
            Vloss_list.append(Vloss)
            Loss = {}
            Loss[u'Tloss'], Loss[u'Vloss'] = Tloss_list, Vloss_list
            hdf5storage.write(Loss, '.', self.log_dir+'/Loss.mat',
                                  store_python_metadata=False, matlab_compatible=True)

            message = 'PSNR: aerial %s, crash %s, drop %s, kobe %s, runner %s, traffic %s' % (psnr_aerial, psnr_crash, psnr_drop, psnr_kobe, psnr_runner, psnr_traffic)
            self.logger.info(message)
            
            if Vloss <= min_val_loss:
                model_filename = self.save_model(self.saver, epoch_cnt, Vloss)
                self.logger.info('Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss,Vloss, model_filename))
                min_val_loss,wait = Vloss,0
            else:
                wait += 1
                if wait > self.patience:
                    model_filename = self.save_model(self.saver, epoch_cnt, Vloss)
                    self.logger.info('Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss,Vloss, model_filename))
                    self.logger.warn('Early stopping at epoch: %d' % epoch_cnt)
                    break
            
            epoch_cnt += 1
            sys.stdout.flush()
            
    def test(self):
        
        print ("Testing Started")
        
        self.restore()
        start_time = time.time()
        test_fetches = {'global_step': tf.train.get_or_create_global_step(),
                        'pred_orig':   self.Decoder_valid.decoded_image
                        #'metrics':     self.Decoder_test.metrics,'loss':        self.Decoder_test.loss
                       }
        
        list_t, list_p, list_m = [], [], []
        matcontent_v = {}
        for tested_batch in range(self.test_size):
            start1 = time.time()
            # (measure_test,mask_train,ground_test) = self.gen_test.next()
            (measure_test,mask_train,ground_test) = self.gen_test.__next__() # zzh
            print('One batch loaded in %s' % (time.time()-start1))
            start2 = time.time()
            feed_dict_test = {self.meas_sample: measure_test,self.sense_matrix: mask_train,self.truth_seg: ground_test}
            test_output = self.sess.run(test_fetches,feed_dict=feed_dict_test)
            print('One batch reconstructed in %s' % (time.time()-start2))
                
            list_t.append(ground_test)
            list_p.append(test_output['pred_orig'])
            list_m.append(measure_test)
        matcontent_v[u'pred'],matcontent_v[u'meas'], matcontent_v[u'truth']= list_p, list_m, list_t
        hdf5storage.write(matcontent_v,'.',self.log_dir+'/Test_result.mat',
                          store_python_metadata=False,matlab_compatible=True)
        print("Testing Finished")
    
    
        
    def calculate_scheduled_lr(self, epoch, min_lr=1e-8):
        decay_factor = int(math.ceil((epoch - self.lr_decay_epoch) / float(self.lr_decay_interval)))
        new_lr = self.lr_init * (self.lr_decay_coe ** max(0, decay_factor))
        new_lr = max(min_lr, new_lr)
        
        self.logger.info('Current learning rate to: %.6f' % new_lr)
        sys.stdout.flush()
        
        self.sess.run(self.lr_update, feed_dict={self.lr_new: new_lr})
        self.Decoder_train.set_lr(self.learning_rate) 
        return new_lr
