# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 13:18:03 2022

@author: Zhenzi Weng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

supported_rnns = {"lstm": tf.keras.layers.LSTMCell,
                  "rnn": tf.keras.layers.SimpleRNNCell,
                  "gru": tf.keras.layers.GRUCell,}

batch_norm_epsilon = 1e-5
batch_norm_decay = 0.997
conv_filters = 32

def batch_norm(inputs, name):
    normed_input = tf.keras.layers.BatchNormalization(momentum=batch_norm_decay, 
                                                      epsilon=batch_norm_epsilon,
                                                      name=name)(inputs)
    
    return normed_input

def conv_bn_layer(inputs, filters, kernel_size, strides, padding, layer_id):
    inputs = tf.pad(inputs, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    inputs = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                    strides=strides, padding="valid",
                                    use_bias=False, activation=tf.nn.relu6,
                                    name="cnn_{}".format(layer_id))(inputs)
    normed_inputs = batch_norm(inputs, name="cnn_bn_{}".format(layer_id))
    
    return normed_inputs

def rnn_layer(inputs, rnn_cell, rnn_hidden_size, is_bidirectional, layer_id):
    if is_bidirectional:
      rnn_outputs = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(rnn_cell(rnn_hidden_size),
                                                  return_sequences=True),
                                                  name="rnn_{}".format(layer_id))(inputs)
    else:
      rnn_outputs = tf.keras.layers.RNN(rnn_cell(rnn_hidden_size), return_sequences=True,
                                        name="rnn_{}".format(layer_id))(inputs)
    rnn_bn_outputs = batch_norm(rnn_outputs, name="rnn_bn_{}".format(layer_id))
    
    return rnn_bn_outputs

def channel_layer(x_norm, AWGN_flag, Rayleigh_flag, Rician_flag, std):
    batch_size = tf.shape(x_norm)[0]
    num_symbol = tf.shape(x_norm)[1]    
    symbol_dim = x_norm.get_shape().as_list()[2]
    
    x_norm = tf.reshape(x_norm, [batch_size, -1])
    paddings = tf.zeros(shape = [batch_size, 256-num_symbol*symbol_dim%256])
    x_norm = tf.concat([x_norm, paddings], -1)
    x_norm = tf.reshape(x_norm, [batch_size, -1, 128, 2 ])
    
    h = tf.random.normal(shape=[batch_size, tf.shape(x_norm)[1], 1, 2], dtype=tf.float32)
    h = (tf.math.sqrt(Rician_flag[0]) + tf.math.sqrt(Rayleigh_flag[0])*h) / tf.sqrt(2.)
    
    noise = tf.random.normal(shape=tf.shape(x_norm), stddev=std[0], dtype=tf.float32)
    
    hx_real = tf.math.multiply(x_norm[:, :, :, 0], h[:, :, :, 0]+AWGN_flag[0]) - tf.math.multiply(x_norm[:, :, :, 1], h[:, :, :, 1])
    hx_imag = tf.math.multiply(x_norm[:, :, :, 0], h[:, :, :, 1]) + tf.math.multiply(x_norm[:, :, :, 1], h[:, :, :, 0]+AWGN_flag[0])
    
    y_real = hx_real + noise[:, :, :, 0]
    y_imag = hx_imag + noise[:, :, :, 1]
    # perfect CSI
    r_real = tf.math.multiply(y_real, h[:, :, :, 0]+AWGN_flag[0]) + tf.math.multiply(y_imag, h[:, :, :, 1])
    r_imag = tf.math.multiply(y_imag, h[:, :, :, 0]+AWGN_flag[0]) - tf.math.multiply(y_real, h[:, :, :, 1])
    
    hh = tf.math.multiply(h[:, :, :, 0], h[:, :, :, 0]) + tf.math.multiply(h[:, :, :, 1], h[:, :, :, 1])
    
    x_hat_real = tf.math.divide(r_real, hh+AWGN_flag[0])
    x_hat_imag = tf.math.divide(r_imag, hh+AWGN_flag[0])
    x_hat_real = tf.reshape(x_hat_real, [batch_size, tf.shape(x_norm)[1], 128, 1])
    x_hat_imag = tf.reshape(x_hat_imag, [batch_size, tf.shape(x_norm)[1], 128, 1])
    x_hat = tf.concat([x_hat_real, x_hat_imag], -1)
    x_hat = tf.reshape(x_hat, [batch_size, -1])
    x_hat = x_hat[:, 0:num_symbol*symbol_dim]
    x_hat = tf.reshape(x_hat, [batch_size, num_symbol, symbol_dim])
    
    return x_hat

class ASR_model(object):
    def __init__(self, args, num_classes):
        self.num_rnn_layers = args.num_rnn_layers
        self.rnn_hidden_size = args.rnn_hidden_size
        self.rnn_type = args.rnn_type
        self.is_bidirectional = args.is_bidirectional
        self.use_bias = args.use_bias
        self.num_classes = num_classes
        self.num_channel_units = args.num_channel_units
        
    def __call__(self, features_inputs, AWGN_flag, Rayleigh_flag, Rician_flag, std):
        ############  semantic coding  ############
        # CNN layer
        cnn_bn_out1 = conv_bn_layer(features_inputs, conv_filters, kernel_size=(41, 11),
                               strides=(2, 2), padding=(20, 5), layer_id=1)
        cnn_bn_out2 = conv_bn_layer(cnn_bn_out1, conv_filters, kernel_size=(21, 11),
                               strides=(2, 1), padding=(10, 5), layer_id=2)
        batch_size = tf.shape(cnn_bn_out2)[0]
        feature_size = cnn_bn_out2.get_shape().as_list()[2]
        rnn_bn_inputs = tf.reshape(cnn_bn_out2, [batch_size, -1, feature_size*conv_filters])
        # RNN layer
        rnn_cell = supported_rnns[self.rnn_type]
        for layer_counter in range(self.num_rnn_layers):
            layer_id = layer_counter + 1
            rnn_bn_inputs = rnn_layer(rnn_bn_inputs, rnn_cell, self.rnn_hidden_size, 
                                      self.is_bidirectional, layer_id)
        # FC layer
        chan_enc_inputs = tf.keras.layers.Dense(self.num_classes, use_bias=self.use_bias,
                                                activation="softmax",
                                                name="dense_{}".format(1))(rnn_bn_inputs)
        ############  channel coding  ############
        # channel encoding
        chan_enc_1 = tf.keras.layers.Dense(self.num_channel_units, use_bias=True,
                                           activation=tf.nn.relu6,
                                           name="enc_{}".format(1))(chan_enc_inputs)
        chan_enc_2 = tf.keras.layers.Dense(self.num_channel_units, use_bias=True,
                                           activation=None,
                                           name="enc_{}".format(2))(chan_enc_1)
        x = tf.reshape(chan_enc_2, [batch_size, -1, 2], name="x")
        # channel layer
        x_norm = tf.math.sqrt(tf.cast(tf.shape(x)[1]/2, tf.float32))*tf.math.l2_normalize(x, axis=1)
        x_hat = channel_layer(x_norm, AWGN_flag, Rayleigh_flag, Rician_flag, std)
        # channel decoding
        chan_dec_inputs = tf.reshape(x_hat, [batch_size, -1, self.num_channel_units])
        chan_dec_1 = tf.keras.layers.Dense(self.num_channel_units, use_bias=True, 
                                           activation=tf.nn.relu6,
                                           name="dec_{}".format(1))(chan_dec_inputs)
        chan_dec_2 = tf.keras.layers.Dense(self.num_channel_units, use_bias=True,
                                           activation=tf.nn.relu6,
                                           name="dec_{}".format(2))(chan_dec_1)
        logits = tf.keras.layers.Dense(self.num_classes, use_bias=True, 
                                       activation="softmax",
                                       name="dense_{}".format(2))(chan_dec_2)
        
        return logits

