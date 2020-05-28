import tensorflow as tf
import numpy as np
import math
import os
import json

from Lib.Utility import *
from Model.Base_TFModel import Basement_TFModel

class MultiDim_Analyzer(Basement_TFModel):
    
    def __init__(self, value_sets, init_learning_rate, sess, config, is_training=True, *args, **kwargs):
        
        super(MultiDim_Analyzer, self).__init__(sess=sess, config=config, learning_rate=init_learning_rate,is_training=is_training)
        '''
        Model input Explanation & Reminder:
        enc_input: the masked input of encoder, the dimension of time may varies with dec_input by [data generation and assignment in handler]
        dec_input: the masked input of encoder, the dimension of time may varies with enc_input by [data generation and assignment in handler]
        truth_pred: used in loss calculation: have identical dimension with decoder input, same(shifted) value for completion(prediction)
        truth_mask: used in the calculation of metric like MSE and relabel the model output for comparison (Mask of Natural Loss)
        shared_info: auxiliary information for encoder/decoder input encoding for identity dimension. BTW, encoding in time dimension is similar to NLP
        train_ave: the average value of all the available data in the training dataset (No usage)
        scalar: the maximum value of each measurement, used for metric calculation
        
        For data in the value_sets (Data Unzip):
        Size for model input/output: (batch_size, num_identity, num_measurement, period_enc/dec, 1)
        Size for auxiliary shared_info: (batch_size, num_shared_feature)
        '''
        (enc_input, dec_input, truth_pred, truth_mask, move_mask, shared_info, train_ave, scalar) = value_sets
        self.num_identity,self.num_measurement = enc_input.get_shape().as_list()[1:3]

        # Initialization of the model hyperparameter, enc-dec structure, evaluation metric & Optimizier
        self.initial_parameter()
        self.model_output = self.encdec_handler(enc_input, dec_input, shared_info)
        self.metric_opt(self.model_output, truth_pred, truth_mask, move_mask, scalar)

    def encdec_handler(self, enc_input, dec_input, shared_info):
        # for dimention introduction of the input: refer to class initialization
        
        # Options for Model Structure
        # Independent -------------- 3 Enc-Dec model are used to learn the relationship for each dimension repectively
        # Sequence ----------------- Following an arbitrary order to do the multiplication on the value vector
        # Element-wise-addition ---- Multiplication on the same value vector and then do the element-wise addition (average)
        # Concatenation ------------ Multiplication on the same value vector and then do the concatenation and matmul (generalized ew-addition)
        # Dimension-reduce --------- Expand the input 3D value to a 1D vector and then calculate the huge AM. (limitation of memory)
        
        # The encode is available for Identity&Measurement and Time dimension
        (shared_encoder,shared_decoder,time_encoder,time_decoder) = self.auxiliary_encode(shared_info) 
        if self.flag_casuality == True:
            mask_casuality = self.casual_mask()
        else:
            mask_casuality = None

        if self.model_structure == 'Independent':
            # Three Enc-Dec structure are built to learn the relationship between identity, measurement and time independently
            # The output of three decoders are combined together through concatenation and matrix multiplication
            enc_input_id,enc_input_meas,enc_input_time = enc_input,tf.transpose(enc_input, [0,2,3,1,4]),tf.transpose(enc_input, [0,3,1,2,4])
            dec_input_id,dec_input_meas,dec_input_time = dec_input,tf.transpose(dec_input, [0,2,3,1,4]),tf.transpose(dec_input, [0,3,1,2,4])
            
            with tf.variable_scope('Identity'):
                if self.flag_identity == True:
                    enc_input_id = enc_input_id + shared_encoder
                    dec_input_id = dec_input_id + shared_decoder
                topenc_id = self.encoder(enc_input_id, self.model_structure)
                attout_id = self.decoder(dec_input_id, self.model_structure, topenc_id)
            with tf.variable_scope('Measurement'):
                topenc_meas = self.encoder(enc_input_meas, self.model_structure)
                attout_meas = self.decoder(dec_input_meas, self.model_structure, topenc_meas)
            with tf.variable_scope('Time'):
                if self.flag_time == True:
                    enc_input_time = enc_input_time + tf.transpose(time_encoder, [0,3,1,2,4])
                    dec_input_time = dec_input_time + tf.transpose(time_decoder, [0,3,1,2,4])
                topenc_time = self.encoder(enc_input_time, self.model_structure)
                attout_time = self.decoder(dec_input_time, self.model_structure, topenc_time, mask_casuality)
            AttOut_Concat = tf.concat([attout_id,tf.transpose(attout_meas,perm=[0,3,1,2,4]),tf.transpose(attout_time,perm=[0,2,3,1,4])], 4)
            return tf.layers.dense(AttOut_Concat, 1, name='Independent_Combine')
        else:
            if self.flag_time == True:
                enc_input = enc_input + time_encoder
                enc_input = enc_input + time_decoder
            if self.flag_identity == True:
                enc_input = enc_input + shared_encoder
                enc_input = enc_input + shared_decoder
            topenc = self.encoder(enc_input, self.model_structure)
            attout = self.decoder(dec_input, self.model_structure, topenc, mask_casuality)
            return attout
        
    def encoder(self,enclayer_init,model_structure):
        enclayer_in = enclayer_init# + self.pos_encoded
        with tf.variable_scope('Encoder'):
            for cnt_enclayer in range(0,self.num_enclayer):
                with tf.variable_scope('layer_%d'%(cnt_enclayer)):
                    enclayer_in = self.layer_norm(enclayer_in + self.multihead_attention(
                        enclayer_in, self.attention_unit, model_structure), 'norm_1')
                    enclayer_in = self.layer_norm(enclayer_in + self.feed_forward_layer(
                        enclayer_in, self.conv_unit, self.filter_encdec), 'norm_2')#
            return enclayer_in

    def decoder(self, declayer_init, model_structure, encoder_top, mask_casuality=None):
        declayer_in = declayer_init
        with tf.variable_scope('Decoder'):
            with tf.variable_scope('layer_0'):
                declayer_in = self.layer_norm(declayer_in + self.multihead_attention(
                    declayer_in, self.attention_unit, model_structure, mask=mask_casuality), 'norm_1')
                (attention_out,KVtop_share) = self.multihead_attention(
                    declayer_in, self.attention_unit, model_structure, top_encod=encoder_top, scope='enc-dec-attention')
                declayer_in = self.layer_norm(declayer_in + attention_out, 'norm_2')
                declayer_in = self.layer_norm(declayer_in + self.feed_forward_layer(
                    declayer_in, self.conv_unit, self.filter_encdec), 'norm_3')
            for cnt_declayer in range(1,self.num_declayer):
                with tf.variable_scope('layer_%d'%(cnt_declayer)):
                    declayer_in = self.layer_norm(declayer_in + self.multihead_attention(
                        declayer_in, self.attention_unit, model_structure, mask=mask_casuality), 'norm_1')
                    declayer_in = self.layer_norm(declayer_in + self.multihead_attention(
                        declayer_in, self.attention_unit, model_structure, top_encod=encoder_top, cache=KVtop_share, 
                        scope='enc-dec-attention'), 'norm_2')
                    declayer_in = self.layer_norm(declayer_in + self.feed_forward_layer(
                        declayer_in, self.conv_unit, self.filter_encdec), 'norm_3')
        with tf.variable_scope('Dec_pred'):
            return self.feed_forward_layer(declayer_in, self.pred_unit, self.filter_pred)

    def metric_opt(self,model_output, truth_pred, truth_mask, move_mask, scalar):

        global_step = tf.train.get_or_create_global_step()
        avail_output = tf.multiply(model_output,truth_mask)
        avail_truth = tf.multiply(truth_pred,truth_mask)
        
        if self.loss_func == 'MSE':
            self.loss = loss_mse(avail_output, avail_truth, truth_mask)
        elif self.loss_func == 'RMSE':
            self.loss = loss_rmse(avail_output, avail_truth, truth_mask)
        else:
            self.loss = loss_mse(avail_output, avail_truth, truth_mask)
            
        orig_output = tf.multiply(model_output,truth_mask-move_mask)
        orig_truth = tf.multiply(truth_pred,truth_mask-move_mask)  
        self.metrics = calculate_metrics(orig_output, orig_truth, scalar, truth_mask-move_mask)

        # Not Sure about the Function
        if self.is_training:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
        self.info_merge = tf.summary.merge_all()
        

    def multihead_attention(self, att_input, att_unit, model_structure, top_encod=None, cache=None, mask=None, scope='self-attention'):
        """
        att_input: the input to be calculated in this module with size of [batch, #identity, #measurement, time, 1]
        att_unit:  the hyperparameter for the dimention of Q/K and V
        top_encod: the output from the top encoder layer [batch_size,#identity, #measurement, time, 1]
        mask: mask the casuality of the self-attention layer, [batch, time, time] or [batch, #id*#meas*time, #id*#meas*time]
        
        3D convolution is applied to realize the unit expansion. For the convenience of application, we have the following index mapping:
        [batch, in_depth, in_height, in_width, in_channels] = [batch_size, num_identity, num_measurement, length_time, 1]
        """
        # Initialization for some necessary item
        (value_units, weight_units) = att_unit
        KVtop_cache = None
        # Since the value of num_measurement and period are small and may equal to each other by coincidence
        # We use the dimention of num_identity
        if model_structure == 'Independent':
            if att_input.get_shape().as_list()[3] == self.num_identity:
                AM_filters, AM_kernal, AM_stride = weight_units*self.num_heads, (1, self.AM_timfuse, 1), (1, self.AM_timjump, 1)
                V_filters, V_kernal, V_stride = value_units*self.num_heads, (1, self.V_timfuse, 1), (1, self.V_timjump, 1)
            elif att_input.get_shape().as_list()[2] == self.num_identity:
                AM_filters, AM_kernal, AM_stride = weight_units*self.num_heads, (self.AM_timfuse, 1, 1), (self.AM_timjump, 1, 1)
                V_filters, V_kernal, V_stride = value_units*self.num_heads, (self.V_timfuse, 1, 1), (self.V_timjump, 1, 1)
            else:
                AM_filters, AM_kernal, AM_stride = weight_units*self.num_heads, (1, 1, self.AM_timfuse), (1, 1, self.AM_timjump)
                V_filters, V_kernal, V_stride = value_units*self.num_heads, (1, 1, self.V_timfuse), (1, 1, self.V_timjump)
        else:
            AM_filters, AM_kernal, AM_stride = weight_units*self.num_heads, (1, 1, self.AM_timfuse), (1, 1, self.AM_timjump)
            V_filters, V_kernal, V_stride = value_units*self.num_heads, (1, 1, self.V_timfuse), (1, 1, self.V_timjump)
        
        
        with tf.variable_scope(scope):
            if top_encod is None or cache is None:
                if top_encod is None:
                    top_encod = att_input
                else:
                    if cache is None:
                        KVtop_cache = {}
                # Linear projection (unit expansion) for multihead-attention dimension: 
                # [self.batch_size, self.num_identity, self.num_measurement, self.length_time, 'hidden-units'*self.num_heads]
                Q = tf.layers.conv3d(inputs=att_input, filters=AM_filters, kernel_size=AM_kernal, strides=AM_stride, 
                                     padding="same", data_format="channels_last", name='Q')
                K = tf.layers.conv3d(inputs=top_encod, filters=AM_filters, kernel_size=AM_kernal, strides=AM_stride, 
                                     padding="same", data_format="channels_last", name='K')
                V = tf.layers.conv3d(inputs=top_encod, filters=V_filters, kernel_size=V_kernal, strides=V_stride, 
                                     padding="same", data_format="channels_last", name='V')
                if KVtop_cache is not None:
                    KVtop_cache = {'share_K':K, 'share_V':V}
            else:
                Q = tf.layers.conv3d(inputs=att_input, filters=AM_filters, kernel_size=AM_kernal, strides=AM_stride, 
                                     padding="same", data_format="channels_last", name='Q')
                K,V = cache['share_K'],cache['share_V']

            # Split the matrix to multiple heads and then concatenate to build a larger batch size: 
            # [self.batch_size*self.num_heads, self.num_identity, self.num_measurement, self.length_time, 'hidden-units']
            Q_headbatch = tf.concat(tf.split(Q, self.num_heads, axis=4), axis=0)
            K_headbatch = tf.concat(tf.split(K, self.num_heads, axis=4), axis=0)
            V_headbatch = tf.concat(tf.split(V, self.num_heads, axis=4), axis=0)
            
            if mask is not None:
                if model_structure == 'Dimension-reduce':
                    recur_time = self.num_identity*self.num_measurement
                    mask_recur = tf.tile(mask, [self.num_heads, recur_time, recur_time])
                else:
                    mask_recur = tf.tile(mask, [self.num_heads, 1, 1])
            else:
                mask_recur = None
            
            out = self.softmax_combination(Q_headbatch, K_headbatch, V_headbatch, model_structure, mask_recur)

            # Merge the multi-head back to the original shape 
            # [batch_size, self.num_identity, self.num_measurement, self.length_time, 'hidden-units'*self.num_heads]
            out = tf.concat(tf.split(out, self.num_heads, axis=0), axis=4)  # 
            out = tf.layers.dense(out, 1, name='multihead_fuse')
            out = tf.layers.dropout(out, rate=self.attdrop_rate, training=self.is_training)
            
            if KVtop_cache is None:
                return out
            else:
                return (out,KVtop_cache)
    
    def feed_forward_layer(self, info_attention, num_hunits, filter_type='dense'):
        '''
        forward_type: 
        "dense" indicates dense layer, 
        "graph" indicates graph based FIR filter (graph convolution),
        "attention" indicates applying the attention algorithm
        "conv" indicates the shared convolution kernal is applied instead of a big weight matrix
        self.ffndrop_rate may be considered later 03122019
        '''
        channel = info_attention.get_shape().as_list()[-1]
        if filter_type == 'dense':
            ffn_dense = tf.layers.dense(info_attention, num_hunits, use_bias=True, activation=tf.nn.relu, name='dense_1')
            return tf.layers.dense(ffn_dense, channel, use_bias=True, name='dense_2')
        elif filter_type == 'graph': 
            raise NotImplementedError
        elif filter_type == 'attention':
            raise NotImplementedError
        elif filter_type == 'conv':
            raise NotImplementedError
    
    def layer_norm(self, norm_input, name_stage):
        
        tfrescale_list = []
        if norm_input.get_shape().as_list()[1] == self.num_identity:
            for num_mea in range(self.num_measurement):
                tfrescale_list.append(tf.contrib.layers.layer_norm(norm_input[:,:,num_mea,:,:], center=True, scale=True,
                                                                   scope=name_stage+str(num_mea)))
            tfrescaled = tf.stack(tfrescale_list)
            return tf.transpose(tfrescaled,perm=[1,2,0,3,4])
        elif norm_input.get_shape().as_list()[2] == self.num_identity:
            for num_mea in range(self.num_measurement):
                tfrescale_list.append(tf.contrib.layers.layer_norm(norm_input[:,:,:,num_mea,:], center=True, scale=True,
                                                                   scope=name_stage+str(num_mea)))
            tfrescaled = tf.stack(tfrescale_list)
            return tf.transpose(tfrescaled,perm=[1,2,3,0,4])
        elif norm_input.get_shape().as_list()[3] == self.num_identity:
            for num_mea in range(self.num_measurement):
                tfrescale_list.append(tf.contrib.layers.layer_norm(norm_input[:,num_mea,:,:,:], center=True, scale=True,
                                                                   scope=name_stage+str(num_mea)))
            tfrescaled = tf.stack(tfrescale_list)
            return tf.transpose(tfrescaled,perm=[1,0,2,3,4])

    def softmax_combination(self, Q, K, V, model_structure, mask=None):
        '''mask is applied before the softmax layer, no dropout is applied, '''
        weight_units,value_units = Q.get_shape().as_list()[-1],V.get_shape().as_list()[-1]
        segs,ids,meas,time = Q.get_shape().as_list()[:4]
        
        if model_structure == 'Independent':
            
            Q = tf.reshape(Q, [segs, ids, -1])
            K = tf.reshape(K, [segs, ids, -1])
            V = tf.reshape(V, [segs, ids, -1])
            # Check the dimension consistency of the reshaped matrix
            assert Q.get_shape().as_list()[-1] == K.get_shape().as_list()[-1]
            dim_model = Q.get_shape().as_list()[-1]
            
            AttentionMap = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(tf.cast(dim_model, tf.float32))
            if mask is not None:
                AttentionMap = tf.multiply(AttentionMap,mask) + tf.constant(-1.0e10)*(tf.constant(1.0)-mask)
            AttentionMap = tf.nn.softmax(AttentionMap, 2)
            Attention_output = tf.matmul(AttentionMap, V)
            return tf.reshape(Attention_output, [segs, ids, meas, time, value_units])
            
        elif model_structure == 'Dimension-reduce':
            
            Q = tf.reshape(Q, [segs, -1,weight_units])
            K = tf.reshape(K, [segs, -1,weight_units])
            V = tf.reshape(V, [segs, -1,value_units])
            # Check the dimension consistency of the reshaped matrix
            assert Q.get_shape().as_list()[1] == K.get_shape().as_list()[1]
            
            AM_all = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(tf.cast(weight_units, tf.float32))
            if mask is not None:
                AM_all = tf.multiply(AM_all,mask) + tf.constant(-1.0e10)*(tf.constant(1.0)-mask)
            AM_all = tf.nn.softmax(AM_all, 2)
            Attention_output = tf.matmul(AM_all, V)
            return tf.reshape(Attention_output, [segs, ids, meas, time, value_units])
        
        else:
            
            # in the future: how to combine, using learnable weight
            Q_I,K_I = tf.reshape(Q,[segs,ids,-1]),tf.reshape(K,[segs,ids,-1])
            Q_M,K_M = tf.reshape(tf.transpose(Q,[0,2,3,1,4]),[segs,meas,-1]),tf.reshape(tf.transpose(K,[0,2,3,1,4]),[segs,meas,-1])
            Q_T,K_T = tf.reshape(tf.transpose(Q,[0,3,1,2,4]),[segs,time,-1]),tf.reshape(tf.transpose(K,[0,3,1,2,4]),[segs,time,-1])
            #Q_I,K_I = tf.reduce_mean(tf.reduce_mean(Q,3),2),tf.reduce_mean(tf.reduce_mean(K,3),2)
            #Q_M,K_M = tf.reduce_mean(tf.reduce_mean(Q,3),1),tf.reduce_mean(tf.reduce_mean(K,3),1)
            #Q_T,K_T = tf.reduce_mean(tf.reduce_mean(Q,2),1),tf.reduce_mean(tf.reduce_mean(K,2),1)
            
            # Check the dimension consistency of the combined matrix
            assert Q_I.get_shape().as_list()[1:] == K_I.get_shape().as_list()[1:]
            assert Q_M.get_shape().as_list()[1:] == K_M.get_shape().as_list()[1:]
            assert Q_T.get_shape().as_list()[1:] == K_T.get_shape().as_list()[1:]
            
            # Build the Attention Map
            AM_Identity = tf.matmul(Q_I, tf.transpose(K_I, [0, 2, 1])) / tf.sqrt(tf.cast(weight_units, tf.float32))
            AM_Measure = tf.matmul(Q_M, tf.transpose(K_M, [0, 2, 1])) / tf.sqrt(tf.cast(weight_units, tf.float32))
            AM_Time = tf.matmul(Q_T, tf.transpose(K_T, [0, 2, 1])) / tf.sqrt(tf.cast(weight_units, tf.float32))
            if mask is not None:
                AM_Time = tf.multiply(AM_Time,mask) + tf.constant(-1.0e10)*(tf.constant(1.0)-mask)
            AM_Identity = tf.nn.softmax(AM_Identity, 2)
            AM_Measure = tf.nn.softmax(AM_Measure, 2)
            AM_Time = tf.nn.softmax(AM_Time, 2)
            
            shape_id = [segs, ids, meas, time, value_units]
            shape_meas = [segs, meas, time, ids, value_units]
            shape_time = [segs, time, ids, meas, value_units]
            
            if model_structure == 'Sequence':
                Out_Id = tf.reshape(tf.matmul(AM_Identity, tf.reshape(V,[segs, ids, -1])), shape_id)
                Out_Id = tf.transpose(Out_Id,perm=[0,2,3,1,4])
                Out_Id_Meas = tf.reshape(tf.matmul(AM_Measure, tf.reshape(Out_Id,[segs, meas, -1])), shape_meas)
                Out_Id_Meas = tf.transpose(Out_Id_Meas,perm=[0,2,3,1,4])
                Out_Id_Meas_Time = tf.reshape(tf.matmul(AM_Time, tf.reshape(Out_Id_Meas,[segs, time, -1])), shape_time)
                return tf.transpose(Out_Id_Meas_Time,perm=[0,2,3,1,4])
            else:
                V_id,V_meas,V_time = V,tf.transpose(V,perm=[0,2,3,1,4]),tf.transpose(V,perm=[0,3,1,2,4])
                Out_Identity = tf.reshape(tf.matmul(AM_Identity, tf.reshape(V_id,[segs, ids, -1])), shape_id)
                Out_Measure = tf.reshape(tf.matmul(AM_Measure, tf.reshape(V_meas,[segs, meas, -1])), shape_meas)
                Out_Time = tf.reshape(tf.matmul(AM_Time, tf.reshape(V_time,[segs, time, -1])), shape_time)
                
                Out_Measure = tf.transpose(Out_Measure,perm=[0,3,1,2,4])
                Out_Time = tf.transpose(Out_Time,perm=[0,2,3,1,4])
                if model_structure == 'Element-wise-addition':
                    return tf.divide(tf.add(tf.add(Out_Identity,Out_Measure),Out_Time),tf.constant(3.0))
                elif model_structure == 'Concatenation':
                    Attention_output = tf.concat([Out_Identity, Out_Measure, Out_Time], 4)
                    return tf.layers.dense(Attention_output, value_units)
                else:
                    raise UnavailableStructureMode
                    
    def casual_mask(self):
        '''
        This function is only applied in the self-attention layer of decoder.
        The lower triangular matrix is used to indicate the available reference of all position in each calculation
        Key Idea: Only the previous position is applied to predict the future
        '''
        batch_size,period = self.batch_size,self.period_dec
        casual_unit = np.tril(np.ones((period, period)))
        casual_tensor = tf.convert_to_tensor(casual_unit, dtype=tf.float32)
        return tf.tile(tf.expand_dims(casual_tensor, 0), [batch_size, 1, 1])

    def initial_parameter(self):

        config = self.config
        # Parameter Initialization of Data Assignment
        self.batch_size = int(config.get('batch_size',1))
        self.period_enc = int(config.get('period_enc',12))
        self.period_dec = int(config.get('period_dec',12))
        
        # Parameter Initialization of Model Framework
        self.num_heads = int(config.get('num_heads',8))
        self.num_enclayer = int(config.get('num_enclayer',5))
        self.num_declayer = int(config.get('num_declayer',5))
        self.model_structure = self.config.get('model_structure')
        
        # Parameter Initialization of Attention (Q K V)
        self.AM_timjump = int(config.get('time_stride_AM',1))
        self.V_timjump = int(config.get('time_stride_V',1))
        self.AM_timfuse = int(config.get('time_fuse_AM',1))
        self.V_timfuse = int(config.get('time_fuse_V',1))
        vunits,wunits = int(config.get('units_value',6)),int(config.get('units_weight',6))
        self.attention_unit = (vunits, wunits)
        
        # Parameter Initialization of Filter (Enc-Dec, Prediction)
        self.filter_encdec = config.get('filter_encdec','dense')
        self.conv_unit = int(config.get('units_conv',4))
        self.attdrop_rate = float(config.get('drop_rate_attention',0.0))
        self.ffndrop_rate = float(config.get('drop_rate_forward',0.1))
        self.filter_pred = config.get('filter_pred','dense')
        self.pred_unit = int(config.get('units_pred',8))
        
        # label of mask
        self.flag_identity = config.get('flag_identity',False)
        self.flag_time = config.get('flag_time',False)
        self.flag_casuality = config.get('flag_casuality',False)

    def auxiliary_encode(self,shared_info):
        # The concatenation is not applicable in this part since all the attention of all three dimension need to be learned.
        # Expanding each dimention will not make sense for our model.
        # Concatenation with the feature dimention (expanded as 1) is equivalent with the element-wise addition.
        with tf.variable_scope('shared_feature'):
            shared_encoder = tf.layers.dense(tf.expand_dims(shared_info,0), self.period_enc*self.num_measurement, 
                                             use_bias=False, activation=None, name='encoder')
            shared_encoder = tf.reshape(shared_encoder, [1, self.num_identity, self.num_measurement, self.period_enc, 1])
            shared_encoder = tf.tile(shared_encoder, [self.batch_size, 1, 1, 1, 1])
            shared_decoder = tf.layers.dense(tf.expand_dims(shared_info,0), self.period_dec*self.num_measurement, 
                                             use_bias=False, activation=None, name='decoder')
            shared_decoder = tf.reshape(shared_decoder, [1, self.num_identity, self.num_measurement, self.period_dec, 1])
            shared_decoder = tf.tile(shared_decoder, [self.batch_size, 1, 1, 1, 1])
            
            time_encoder = None #self.period_enc tf.convert_to_tensor position=measurement
            time_decoder = None #self.period_dec tf.convert_to_tensor position=measurement
            
            for meas_unit in range(int(self.num_measurement/2)):
                denom = tf.pow(tf.constant(10000.0),tf.constant(2.0*meas_unit/self.num_measurement))
                phase_enc = tf.linspace(0.0,self.period_enc-1.0,self.period_enc)*tf.constant(math.pi/180.0)/denom
                phase_dec = tf.linspace(0.0,self.period_dec-1.0,self.period_dec)*tf.constant(math.pi/180.0)/denom
                sin_enc,cos_enc = tf.expand_dims(tf.sin(phase_enc),0),tf.expand_dims(tf.cos(phase_enc),0)
                sin_dec,cos_dec = tf.expand_dims(tf.sin(phase_dec),0),tf.expand_dims(tf.cos(phase_dec),0)
                if time_encoder is None:
                    time_encoder = tf.concat([sin_enc,cos_enc],0)
                else:
                    time_encoder = tf.concat([time_encoder,sin_enc,cos_enc],0)
                if time_decoder is None:
                    time_decoder = tf.concat([sin_dec,cos_dec],0)
                else:
                    time_decoder = tf.concat([time_decoder,sin_dec,cos_dec],0)
            time_encoder = tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(time_encoder,0),0),[self.batch_size,self.num_identity,1,1]),-1)
            time_decoder = tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(time_decoder,0),0),[self.batch_size,self.num_identity,1,1]),-1)
            return (shared_encoder,shared_decoder,time_encoder,time_decoder)
