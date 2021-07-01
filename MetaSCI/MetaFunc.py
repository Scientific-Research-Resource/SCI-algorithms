

import tensorflow as tf
import numpy as np
#import tensorflow.compat.v1 as tf


def construct_weights(sigmaInit, num_frame):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=sigmaInit, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    weights = {'w1': weight_variable([5, 5, num_frame+1, 32]), 'b1': bias_variable([32]),
               'w2': weight_variable([3, 3, 32, 64]), 'b2': bias_variable([64]),
               'w3': weight_variable([1, 1, 64, 64]), 'b3': bias_variable([64]),
               'w4': weight_variable([3, 3, 64, 128]), 'b4': bias_variable([128]),
               'w7': weight_variable([3, 3, 128, 128]), 'b7': bias_variable([128]),
               'w8': weight_variable([1, 1, 128, 128]), 'b8': bias_variable([128]),
               'w9': weight_variable([3, 3, 128, 128]), 'b9': bias_variable([128]),
               'w10': weight_variable([1, 1, 128, 128]), 'b10': bias_variable([128]),
               'w5': weight_variable([3, 3, 64, 128]), 'b5': bias_variable([64]),
               'w51': weight_variable([3, 3, 64, 32]), 'b51': bias_variable([32]),
               'w52': weight_variable([1, 1, 32, 16]), 'b52': bias_variable([16]),
               'w6': weight_variable([3, 3, 16, num_frame]), 'b6': bias_variable([num_frame]),
               'w_1res1': weight_variable([3, 3, 128, 128]), 'b_1res1': bias_variable([128]),
               'w_1res2': weight_variable([1, 1, 128, 128]), 'b_1res2': bias_variable([128]),
               'w_1res3': weight_variable([3, 3, 128, 128]), 'b_1res3': bias_variable([128]),
               'w_1res4': weight_variable([3, 3, 128, 128]), 'b_1res4': bias_variable([128]),
               'w_1res5': weight_variable([1, 1, 128, 128]), 'b_1res5': bias_variable([128]),
               'w_1res6': weight_variable([3, 3, 128, 128]), 'b_1res6': bias_variable([128]),
               'w_1res7': weight_variable([3, 3, 128, 128]), 'b_1res7': bias_variable([128]),
               'w_1res8': weight_variable([1, 1, 128, 128]), 'b_1res8': bias_variable([128]),
               'w_1res9': weight_variable([3, 3, 128, 128]), 'b_1res9': bias_variable([128]),
               'w_2res1': weight_variable([3, 3, 128, 128]), 'b_2res1': bias_variable([128]),
               'w_2res2': weight_variable([1, 1, 128, 128]), 'b_2res2': bias_variable([128]),
               'w_2res3': weight_variable([3, 3, 128, 128]), 'b_2res3': bias_variable([128]),
               'w_2res4': weight_variable([3, 3, 128, 128]), 'b_2res4': bias_variable([128]),
               'w_2res5': weight_variable([1, 1, 128, 128]), 'b_2res5': bias_variable([128]),
               'w_2res6': weight_variable([3, 3, 128, 128]), 'b_2res6': bias_variable([128]),
               'w_2res7': weight_variable([3, 3, 128, 128]), 'b_2res7': bias_variable([128]),
               'w_2res8': weight_variable([1, 1, 128, 128]), 'b_2res8': bias_variable([128]),
               'w_2res9': weight_variable([3, 3, 128, 128]), 'b_2res9': bias_variable([128]),
               'w_3res1': weight_variable([3, 3, 128, 128]), 'b_3res1': bias_variable([128]),
               'w_3res2': weight_variable([1, 1, 128, 128]), 'b_3res2': bias_variable([128]),
               'w_3res3': weight_variable([3, 3, 128, 128]), 'b_3res3': bias_variable([128]),
               'w_3res4': weight_variable([3, 3, 128, 128]), 'b_3res4': bias_variable([128]),
               'w_3res5': weight_variable([1, 1, 128, 128]), 'b_3res5': bias_variable([128]),
               'w_3res6': weight_variable([3, 3, 128, 128]), 'b_3res6': bias_variable([128]),
               'w_3res7': weight_variable([3, 3, 128, 128]), 'b_3res7': bias_variable([128]),
               'w_3res8': weight_variable([1, 1, 128, 128]), 'b_3res8': bias_variable([128]),
               'w_3res9': weight_variable([3, 3, 128, 128]), 'b_3res9': bias_variable([128])}
    return weights

def construct_weights_modulation(sigmaInit, num_frame): # zzh: modulation params to be adapted
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=sigmaInit, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    def ones_variable(shape):
        initial = tf.constant(1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial)

    weights = {'w1': weight_variable([5, 5, num_frame+1, 32]), 'b1': bias_variable([32]), #'w1': weight_variable([5, 5, 9, 32]), 'b1': bias_variable([32]),
               'w2': weight_variable([3, 3, 32, 64]), 'b2': bias_variable([64]),
               'w3': weight_variable([1, 1, 64, 64]), 'b3': bias_variable([64]),
               'w4': weight_variable([3, 3, 64, 128]), 'b4': bias_variable([128]),
               'w7': weight_variable([3, 3, 128, 128]), 'b7': bias_variable([128]),
               'w8': weight_variable([1, 1, 128, 128]), 'b8': bias_variable([128]),
               'w9': weight_variable([3, 3, 128, 128]), 'b9': bias_variable([128]),
               'w10': weight_variable([1, 1, 128, 128]), 'b10': bias_variable([128]),
               'w5': weight_variable([3, 3, 64, 128]), 'b5': bias_variable([64]),
               'w51': weight_variable([3, 3, 64, 32]), 'b51': bias_variable([32]),
               'w52': weight_variable([1, 1, 32, 16]), 'b52': bias_variable([16]),
               'w6': weight_variable([3, 3, 16, num_frame]), 'b6': bias_variable([num_frame]),
               'w_1res1': weight_variable([3, 3, 128, 128]), 'b_1res1': bias_variable([128]),
               'w_1res2': weight_variable([1, 1, 128, 128]), 'b_1res2': bias_variable([128]),
               'w_1res3': weight_variable([3, 3, 128, 128]), 'b_1res3': bias_variable([128]),
               'w_1res4': weight_variable([3, 3, 128, 128]), 'b_1res4': bias_variable([128]),
               'w_1res5': weight_variable([1, 1, 128, 128]), 'b_1res5': bias_variable([128]),
               'w_1res6': weight_variable([3, 3, 128, 128]), 'b_1res6': bias_variable([128]),
               'w_1res7': weight_variable([3, 3, 128, 128]), 'b_1res7': bias_variable([128]),
               'w_1res8': weight_variable([1, 1, 128, 128]), 'b_1res8': bias_variable([128]),
               'w_1res9': weight_variable([3, 3, 128, 128]), 'b_1res9': bias_variable([128]),
               'w_2res1': weight_variable([3, 3, 128, 128]), 'b_2res1': bias_variable([128]),
               'w_2res2': weight_variable([1, 1, 128, 128]), 'b_2res2': bias_variable([128]),
               'w_2res3': weight_variable([3, 3, 128, 128]), 'b_2res3': bias_variable([128]),
               'w_2res4': weight_variable([3, 3, 128, 128]), 'b_2res4': bias_variable([128]),
               'w_2res5': weight_variable([1, 1, 128, 128]), 'b_2res5': bias_variable([128]),
               'w_2res6': weight_variable([3, 3, 128, 128]), 'b_2res6': bias_variable([128]),
               'w_2res7': weight_variable([3, 3, 128, 128]), 'b_2res7': bias_variable([128]),
               'w_2res8': weight_variable([1, 1, 128, 128]), 'b_2res8': bias_variable([128]),
               'w_2res9': weight_variable([3, 3, 128, 128]), 'b_2res9': bias_variable([128]),
               'w_3res1': weight_variable([3, 3, 128, 128]), 'b_3res1': bias_variable([128]),
               'w_3res2': weight_variable([1, 1, 128, 128]), 'b_3res2': bias_variable([128]),
               'w_3res3': weight_variable([3, 3, 128, 128]), 'b_3res3': bias_variable([128]),
               'w_3res4': weight_variable([3, 3, 128, 128]), 'b_3res4': bias_variable([128]),
               'w_3res5': weight_variable([1, 1, 128, 128]), 'b_3res5': bias_variable([128]),
               'w_3res6': weight_variable([3, 3, 128, 128]), 'b_3res6': bias_variable([128]),
               'w_3res7': weight_variable([3, 3, 128, 128]), 'b_3res7': bias_variable([128]),
               'w_3res8': weight_variable([1, 1, 128, 128]), 'b_3res8': bias_variable([128]),
               'w_3res9': weight_variable([3, 3, 128, 128]), 'b_3res9': bias_variable([128])}

    weights_m = {'w1_L': ones_variable([num_frame+1, 1]), 'w1_R': ones_variable([1, 32]),
               'w2_L': ones_variable([32, 1]), 'w2_R': ones_variable([1, 64]),
               'w3_L': ones_variable([64, 1]), 'w3_R': ones_variable([1, 64]),
               'w4_L': ones_variable([64, 1]), 'w4_R': ones_variable([1, 128]),
               'w7_L': ones_variable([128, 1]), 'w7_R': ones_variable([1, 128]),
               'w8_L': ones_variable([128, 1]), 'w8_R': ones_variable([1, 128]),
               'w9_L': ones_variable([128, 1]), 'w9_R': ones_variable([1, 128]),
               'w10_L': ones_variable([128, 1]), 'w10_R': ones_variable([1, 128]),
               'w5_L': ones_variable([64, 1]), 'w5_R': ones_variable([1, 128]),
               'w51_L': ones_variable([64, 1]), 'w51_R': ones_variable([1, 32]),
               'w52_L': ones_variable([32, 1]), 'w52_R': ones_variable([1, 16]),
               'w6_L': ones_variable([16, 1]), 'w6_R': ones_variable([1, num_frame]),

               'w_1res1_L': ones_variable([128, 1]), 'w_1res1_R': ones_variable([1, 128]),
               'w_1res2_L': ones_variable([128, 1]), 'w_1res2_R': ones_variable([1, 128]),
               'w_1res3_L': ones_variable([128, 1]), 'w_1res3_R': ones_variable([1, 128]),
               'w_1res4_L': ones_variable([128, 1]), 'w_1res4_R': ones_variable([1, 128]),
               'w_1res5_L': ones_variable([128, 1]), 'w_1res5_R': ones_variable([1, 128]),
               'w_1res6_L': ones_variable([128, 1]), 'w_1res6_R': ones_variable([1, 128]),
               'w_1res7_L': ones_variable([128, 1]), 'w_1res7_R': ones_variable([1, 128]),
               'w_1res8_L': ones_variable([128, 1]), 'w_1res8_R': ones_variable([1, 128]),
               'w_1res9_L': ones_variable([128, 1]), 'w_1res9_R': ones_variable([1, 128]),

               'w_2res1_L': ones_variable([128, 1]), 'w_2res1_R': ones_variable([1, 128]),
               'w_2res2_L': ones_variable([128, 1]), 'w_2res2_R': ones_variable([1, 128]),
               'w_2res3_L': ones_variable([128, 1]), 'w_2res3_R': ones_variable([1, 128]),
               'w_2res4_L': ones_variable([128, 1]), 'w_2res4_R': ones_variable([1, 128]),
               'w_2res5_L': ones_variable([128, 1]), 'w_2res5_R': ones_variable([1, 128]),
               'w_2res6_L': ones_variable([128, 1]), 'w_2res6_R': ones_variable([1, 128]),
               'w_2res7_L': ones_variable([128, 1]), 'w_2res7_R': ones_variable([1, 128]),
               'w_2res8_L': ones_variable([128, 1]), 'w_2res8_R': ones_variable([1, 128]),
               'w_2res9_L': ones_variable([128, 1]), 'w_2res9_R': ones_variable([1, 128]),

               'w_3res1_L': ones_variable([128, 1]), 'w_3res1_R': ones_variable([1, 128]),
               'w_3res2_L': ones_variable([128, 1]), 'w_3res2_R': ones_variable([1, 128]),
               'w_3res3_L': ones_variable([128, 1]), 'w_3res3_R': ones_variable([1, 128]),
               'w_3res4_L': ones_variable([128, 1]), 'w_3res4_R': ones_variable([1, 128]),
               'w_3res5_L': ones_variable([128, 1]), 'w_3res5_R': ones_variable([1, 128]),
               'w_3res6_L': ones_variable([128, 1]), 'w_3res6_R': ones_variable([1, 128]),
               'w_3res7_L': ones_variable([128, 1]), 'w_3res7_R': ones_variable([1, 128]),
               'w_3res8_L': ones_variable([128, 1]), 'w_3res8_R': ones_variable([1, 128]),
               'w_3res9_L': ones_variable([128, 1]), 'w_3res9_R': ones_variable([1, 128])}
    return weights, weights_m

def forward(mask, meas_re, gt, weights, batch_size, num_frame, img_dim):
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def conv2d_w_stride(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')

    def conv_t_w_stride(x, w, s):
        return tf.nn.conv2d_transpose(x, w, output_shape=s, strides=[1, 2, 2, 1], padding='SAME')

    def res_part1(x, weights):
        h = tf.nn.leaky_relu(conv2d(x, weights['w_1res1']) + weights['b_1res1'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_1res2']) + weights['b_1res2'], alpha=1e-2)

        h = conv2d(h, weights['w_1res3']) + weights['b_1res3']

        x = x + h

        h = tf.nn.leaky_relu(conv2d(x, weights['w_1res4']) + weights['b_1res4'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_1res5']) + weights['b_1res5'], alpha=1e-2)

        h = conv2d(h, weights['w_1res6']) + weights['b_1res6']

        x = x + h

        h = tf.nn.leaky_relu(conv2d(x, weights['w_1res7']) + weights['b_1res7'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_1res8']) + weights['b_1res8'], alpha=1e-2)

        h = conv2d(h, weights['w_1res9']) + weights['b_1res9']

        x = x + h

        return x

    def res_part2(x, weights):
        h = tf.nn.leaky_relu(conv2d(x, weights['w_2res1']) + weights['b_2res1'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_2res2']) + weights['b_2res2'], alpha=1e-2)

        h = conv2d(h, weights['w_2res3']) + weights['b_2res3']

        x = x + h

        h = tf.nn.leaky_relu(conv2d(x, weights['w_2res4']) + weights['b_2res4'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_2res5']) + weights['b_2res5'], alpha=1e-2)

        h = conv2d(h, weights['w_2res6']) + weights['b_2res6']

        x = x + h

        h = tf.nn.leaky_relu(conv2d(x, weights['w_2res7']) + weights['b_2res7'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_2res8']) + weights['b_2res8'], alpha=1e-2)

        h = conv2d(h, weights['w_2res9']) + weights['b_2res9']

        x = x + h

        return x

    def res_part3(x, weights):
        h = tf.nn.leaky_relu(conv2d(x, weights['w_3res1']) + weights['b_3res1'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_3res2']) + weights['b_3res2'], alpha=1e-2)

        h = conv2d(h, weights['w_3res3']) + weights['b_3res3']

        x = x + h

        h = tf.nn.leaky_relu(conv2d(x, weights['w_3res4']) + weights['b_3res4'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_3res5']) + weights['b_3res5'], alpha=1e-2)

        h = conv2d(h, weights['w_3res6']) + weights['b_3res6']

        x = x + h

        h = tf.nn.leaky_relu(conv2d(x, weights['w_3res7']) + weights['b_3res7'], alpha=1e-2)

        h = tf.nn.leaky_relu(conv2d(h, weights['w_3res8']) + weights['b_3res8'], alpha=1e-2)

        h = conv2d(h, weights['w_3res9']) + weights['b_3res9']

        x = x + h

        return x

    maskt = tf.tile(tf.expand_dims(mask, 0), (batch_size, 1, 1, 1))
    maskt = tf.multiply(maskt, tf.tile(meas_re, (1, 1, 1, num_frame)))
    data = tf.concat([meas_re, maskt], axis=3)

    h = tf.nn.leaky_relu(conv2d(data, weights['w1']) + weights['b1'], alpha=1e-2)

    h = tf.nn.leaky_relu(conv2d(h, weights['w2']) + weights['b2'], alpha=1e-2)

    h = tf.nn.leaky_relu(conv2d(h, weights['w3']) + weights['b3'], alpha=1e-2)

    h = tf.nn.leaky_relu(conv2d_w_stride(h, weights['w4']) + weights['b4'], alpha=1e-2)

    h = res_part1(h, weights)

    h = tf.nn.leaky_relu(conv2d(h, weights['w7']) + weights['b7'], alpha=1e-2)

    h = conv2d(h, weights['w8']) + weights['b8']

    h = res_part2(h, weights)

    h = tf.nn.leaky_relu(conv2d(h, weights['w9']) + weights['b9'], alpha=1e-2)

    h = conv2d(h, weights['w10']) + weights['b10']

    h = res_part3(h, weights)

    h = tf.nn.leaky_relu(conv_t_w_stride(h, weights['w5'], s=[batch_size, img_dim, img_dim, 64]) + weights['b5'], alpha=1e-2)

    h = tf.nn.leaky_relu(conv2d(h, weights['w51']) + weights['b51'], alpha=1e-2)

    h = tf.nn.leaky_relu(conv2d(h, weights['w52']) + weights['b52'], alpha=1e-2)

    pred = conv2d(h, weights['w6']) + weights['b6']

    loss = tf.reduce_mean(tf.keras.losses.MSE(pred, gt))

    task_output = {'pred': pred, 'loss': loss}

    return task_output

def forward_modulation(mask, meas_re, gt, weights, weights_m, batch_size, num_frame, image_dim):
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def conv2d_w_stride(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')

    def conv_t_w_stride(x, w, s):
        return tf.nn.conv2d_transpose(x, w, output_shape=s, strides=[1, 2, 2, 1], padding='SAME')

    def res_part1(x, weights, weights_m):
        w_1res1 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res1_L'], weights_m['w_1res1_R']), axis=0), axis=0) * weights['w_1res1']
        h = tf.nn.leaky_relu(conv2d(x, w_1res1) + weights['b_1res1'], alpha=1e-2)

        w_1res2 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res2_L'], weights_m['w_1res2_R']), axis=0), axis=0) * weights['w_1res2']
        h = tf.nn.leaky_relu(conv2d(h, w_1res2) + weights['b_1res2'], alpha=1e-2)

        w_1res3 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res3_L'], weights_m['w_1res3_R']), axis=0), axis=0) * weights['w_1res3']
        h = conv2d(h, w_1res3) + weights['b_1res3']

        x = x + h

        w_1res4 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res4_L'], weights_m['w_1res4_R']), axis=0), axis=0) * weights['w_1res4']
        h = tf.nn.leaky_relu(conv2d(x, w_1res4) + weights['b_1res4'], alpha=1e-2)

        w_1res5 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res5_L'], weights_m['w_1res5_R']), axis=0), axis=0) * weights['w_1res5']
        h = tf.nn.leaky_relu(conv2d(h,  w_1res5) + weights['b_1res5'], alpha=1e-2)

        w_1res6 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res6_L'], weights_m['w_1res6_R']), axis=0), axis=0) * weights['w_1res6']
        h = conv2d(h, w_1res6) + weights['b_1res6']

        x = x + h

        w_1res7 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res7_L'], weights_m['w_1res7_R']), axis=0), axis=0) * weights['w_1res7']
        h = tf.nn.leaky_relu(conv2d(x, w_1res7) + weights['b_1res7'], alpha=1e-2)

        w_1res8 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res8_L'], weights_m['w_1res8_R']), axis=0), axis=0) * weights['w_1res8']
        h = tf.nn.leaky_relu(conv2d(h, w_1res8) + weights['b_1res8'], alpha=1e-2)

        w_1res9 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_1res9_L'], weights_m['w_1res9_R']), axis=0), axis=0) * weights['w_1res9']
        h = conv2d(h, w_1res9) + weights['b_1res9']

        x = x + h

        return x

    def res_part2(x, weights, weights_m):
        w_2res1 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res1_L'], weights_m['w_2res1_R']), axis=0), axis=0) * weights['w_2res1']
        h = tf.nn.leaky_relu(conv2d(x, w_2res1) + weights['b_2res1'], alpha=1e-2)

        w_2res2 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res2_L'], weights_m['w_2res2_R']), axis=0), axis=0) * weights['w_2res2']
        h = tf.nn.leaky_relu(conv2d(h, w_2res2) + weights['b_2res2'], alpha=1e-2)

        w_2res3 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res3_L'], weights_m['w_2res3_R']), axis=0), axis=0) * weights['w_2res3']
        h = conv2d(h, w_2res3) + weights['b_2res3']

        x = x + h

        w_2res4 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res4_L'], weights_m['w_2res4_R']), axis=0), axis=0) * weights['w_2res4']
        h = tf.nn.leaky_relu(conv2d(x, w_2res4) + weights['b_2res4'], alpha=1e-2)

        w_2res5 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res5_L'], weights_m['w_2res5_R']), axis=0), axis=0) * weights['w_2res5']
        h = tf.nn.leaky_relu(conv2d(h, w_2res5) + weights['b_2res5'], alpha=1e-2)

        w_2res6 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res6_L'], weights_m['w_2res6_R']), axis=0), axis=0) * weights['w_2res6']
        h = conv2d(h, w_2res6) + weights['b_2res6']

        x = x + h

        w_2res7 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res7_L'], weights_m['w_2res7_R']), axis=0), axis=0) * weights['w_2res7']
        h = tf.nn.leaky_relu(conv2d(x, w_2res7) + weights['b_2res7'], alpha=1e-2)

        w_2res8 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res8_L'], weights_m['w_2res8_R']), axis=0), axis=0) * weights['w_2res8']
        h = tf.nn.leaky_relu(conv2d(h, w_2res8) + weights['b_2res8'], alpha=1e-2)

        w_2res9 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_2res9_L'], weights_m['w_2res9_R']), axis=0), axis=0) * weights['w_2res9']
        h = conv2d(h, w_2res9) + weights['b_2res9']

        x = x + h

        return x

    def res_part3(x, weights, weights_m):
        w_3res1 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res1_L'], weights_m['w_3res1_R']), axis=0), axis=0) * weights['w_3res1']
        h = tf.nn.leaky_relu(conv2d(x, w_3res1) + weights['b_3res1'], alpha=1e-2)

        w_3res2 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res2_L'], weights_m['w_3res2_R']), axis=0), axis=0) * weights['w_3res2']
        h = tf.nn.leaky_relu(conv2d(h, w_3res2) + weights['b_3res2'], alpha=1e-2)

        w_3res3 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res3_L'], weights_m['w_3res3_R']), axis=0), axis=0) * weights['w_3res3']
        h = conv2d(h, w_3res3) + weights['b_3res3']

        x = x + h

        w_3res4 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res4_L'], weights_m['w_3res4_R']), axis=0), axis=0) * weights['w_3res4']
        h = tf.nn.leaky_relu(conv2d(x, w_3res4) + weights['b_3res4'], alpha=1e-2)

        w_3res5 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res5_L'], weights_m['w_3res5_R']), axis=0), axis=0) * weights['w_3res5']
        h = tf.nn.leaky_relu(conv2d(h, w_3res5) + weights['b_3res5'], alpha=1e-2)

        w_3res6 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res6_L'], weights_m['w_3res6_R']), axis=0), axis=0) * weights['w_3res6']
        h = conv2d(h, w_3res6) + weights['b_3res6']

        x = x + h

        w_3res7 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res7_L'], weights_m['w_3res7_R']), axis=0), axis=0) * weights['w_3res7']
        h = tf.nn.leaky_relu(conv2d(x, w_3res7) + weights['b_3res7'], alpha=1e-2)

        w_3res8 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res8_L'], weights_m['w_3res8_R']), axis=0), axis=0) * weights['w_3res8']
        h = tf.nn.leaky_relu(conv2d(h, w_3res8) + weights['b_3res8'], alpha=1e-2)

        w_3res9 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w_3res9_L'], weights_m['w_3res9_R']), axis=0), axis=0) * weights['w_3res9']
        h = conv2d(h, w_3res9) + weights['b_3res9']

        x = x + h

        return x

    maskt = tf.tile(tf.expand_dims(mask, 0), (batch_size, 1, 1, 1))
    maskt = tf.multiply(maskt, tf.tile(meas_re, (1, 1, 1, num_frame)))
    data = tf.concat([meas_re, maskt], axis=3)

    w1 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w1_L'], weights_m['w1_R']), axis=0), axis=0) * weights['w1']
    h = tf.nn.leaky_relu(conv2d(data, w1) + weights['b1'], alpha=1e-2)

    w2 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w2_L'], weights_m['w2_R']), axis=0), axis=0) * weights['w2']
    h = tf.nn.leaky_relu(conv2d(h, w2) + weights['b2'], alpha=1e-2)

    w3 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w3_L'], weights_m['w3_R']), axis=0), axis=0) * weights['w3']
    h = tf.nn.leaky_relu(conv2d(h, w3) + weights['b3'], alpha=1e-2)

    w4 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w4_L'], weights_m['w4_R']), axis=0), axis=0) * weights['w4']
    h = tf.nn.leaky_relu(conv2d_w_stride(h, w4) + weights['b4'], alpha=1e-2)

    h = res_part1(h, weights, weights_m)

    w7 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w7_L'], weights_m['w7_R']), axis=0), axis=0) * weights['w7']
    h = tf.nn.leaky_relu(conv2d(h, w7) + weights['b7'], alpha=1e-2)

    w8 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w8_L'], weights_m['w8_R']), axis=0), axis=0) * weights['w8']
    h = conv2d(h, w8) + weights['b8']

    h = res_part2(h, weights, weights_m)

    w9 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w9_L'], weights_m['w9_R']), axis=0), axis=0) * weights['w9']
    h = tf.nn.leaky_relu(conv2d(h, w9) + weights['b9'], alpha=1e-2)

    w10 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w10_L'], weights_m['w10_R']), axis=0), axis=0) * weights['w10']
    h = conv2d(h, w10) + weights['b10']

    h = res_part3(h, weights, weights_m)

    w5 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w5_L'], weights_m['w5_R']), axis=0), axis=0) * weights['w5']
    h = tf.nn.leaky_relu(conv_t_w_stride(h, w5, s=[batch_size, image_dim, image_dim, 64]) + weights['b5'], alpha=1e-2)

    w51 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w51_L'], weights_m['w51_R']), axis=0), axis=0) * weights['w51']
    h = tf.nn.leaky_relu(conv2d(h, w51) + weights['b51'], alpha=1e-2)

    w52 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w52_L'], weights_m['w52_R']), axis=0), axis=0) * weights['w52']
    h = tf.nn.leaky_relu(conv2d(h, w52) + weights['b52'], alpha=1e-2)

    w6 = tf.expand_dims(tf.expand_dims(tf.matmul(weights_m['w6_L'], weights_m['w6_R']), axis=0), axis=0) * weights['w6']
    pred = conv2d(h, w6) + weights['b6']

    loss = tf.reduce_mean(tf.keras.losses.MSE(pred, gt))

    task_output = {'pred': pred, 'loss': loss}

    return task_output

def MAML(mask, X_meas_re, X_gt, Y_meas_re, Y_gt, weights, batch_size, num_frame, update_lr, num_updates):
    def every_task(inp):
        mask, X_meas_re, X_gt, Y_meas_re, Y_gt = inp

        xtask_output = forward(mask, X_meas_re, X_gt, weights, batch_size, num_frame, 256)
        grads = tf.gradients(xtask_output['loss'], list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict(zip(weights.keys(), [weights[key] - update_lr * gradients[key] for key in weights.keys()]))

        for j in range(num_updates - 1):
            xtask_output = forward(mask, X_meas_re, X_gt, fast_weights, batch_size, num_frame)
            grads = tf.gradients(xtask_output['loss'], list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - update_lr * gradients[key] for key in fast_weights.keys()]))

        ytask_output = forward(mask, Y_meas_re, Y_gt, fast_weights, batch_size, num_frame)

        return ytask_output['loss']

    inp = [mask, X_meas_re, X_gt, Y_meas_re, Y_gt]
    Loss = tf.map_fn(every_task, elems=inp, dtype=tf.float32)
    Loss = tf.reduce_mean(Loss)

    final_output = {'loss': Loss}

    return final_output

def MAML_parallel(mask, X_meas_re, X_gt, Y_meas_re, Y_gt, weights, batch_size, num_frame, update_lr, num_updates, gpus):

    tower_loss = []
    for i in range(len(gpus)):
        with tf.device('/gpu:%d' % i):
            with tf.variable_scope('forward', reuse=tf.AUTO_REUSE):
                xtask_output = forward(mask[i], X_meas_re[i], X_gt[i], weights, batch_size, num_frame)
                grads = tf.gradients(xtask_output['loss'], list(weights.values()))
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - update_lr * gradients[key] for key in weights.keys()]))

                for j in range(num_updates - 1):
                    xtask_output = forward(mask[i], X_meas_re[i], X_gt[i], fast_weights, batch_size, num_frame)
                    grads = tf.gradients(xtask_output['loss'], list(fast_weights.values()))
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - update_lr * gradients[key] for key in fast_weights.keys()]))

                ytask_output = forward(mask[i], Y_meas_re[i], Y_gt[i], fast_weights, batch_size, num_frame)

                loss_op = ytask_output['loss']

    Loss = tf.reduce_mean(tower_loss)

    final_output = {'loss': Loss}

    return final_output

def MAML_modulation(mask, X_meas_re, X_gt, Y_meas_re, Y_gt, weights, weights_m, batch_size, num_frame, image_dim, update_lr, num_updates):
    def every_task(inp):
        mask, X_meas_re, X_gt, Y_meas_re, Y_gt = inp

        xtask_output = forward_modulation(mask, X_meas_re, X_gt, weights, weights_m, batch_size, num_frame, image_dim)
        grads = tf.gradients(xtask_output['loss'], list(weights_m.values()))
        gradients = dict(zip(weights_m.keys(), grads))
        fast_weights = dict(zip(weights_m.keys(), [weights_m[key] - update_lr * gradients[key] for key in weights_m.keys()]))

        for j in range(num_updates - 1):
            xtask_output = forward_modulation(mask, X_meas_re, X_gt, weights, fast_weights, batch_size, num_frame, image_dim)
            grads = tf.gradients(xtask_output['loss'], list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - update_lr * gradients[key] for key in fast_weights.keys()]))

        ytask_output = forward_modulation(mask, Y_meas_re, Y_gt, weights, fast_weights, batch_size, num_frame, image_dim)

        return ytask_output['loss'], ytask_output['pred']
    
               

    # Loss & Pred
    inp = [mask, X_meas_re, X_gt, Y_meas_re, Y_gt]
    
    # for each task: do 'every_task' function (self-designed manual grad descent for 'weights_m')
    MAML_out = tf.map_fn(every_task, elems=inp, dtype=(tf.float32,tf.float32))
        
    Loss = tf.reduce_mean(MAML_out[0])
    Pred = MAML_out[1]

    final_output = {'loss': Loss, 'pred':Pred}

    return final_output
