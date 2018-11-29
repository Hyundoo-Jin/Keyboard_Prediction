import tensorflow as tf
import numpy as np
import os, sys
import time
from tqdm import tqdm
import time
import pickle

class Predict_Model() :
    def __init__(self, arguments) :
        self.argument_setting(arguments)

    def argument_setting(self, arguments) :
        # set model type
        if arguments.model == 'l' or arguments.model == 'lstm' :
            self.modeltype = 'lstm'
            self.batch_size = arguments.batch_size
            self.learning_rate = arguments.learning_rate
            self.epochs = arguments.epoch
        elif arguments.model == 'c' or arguments.model == 'cnn' :
            self.modeltype = 'cnn'
            self.batch_size = arguments.batch_size
            self.learning_rate = arguments.learning_rate
            self.epochs = arguments.epoch
        elif arguments.model == 'm' or arguments.model == 'ml' :
            self.modeltype = 'ml'
        else :
            raise AttributeError('Undefined model type.')
        
        # set 
    
    def _weight_variable(self, layer_name, shape):
        return tf.get_variable(layer_name + "_w", shape = shape,
            initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01))

    def _bias_variable(self, layer_name, shape) :
        return tf.get_variable(layer_name + '_b', shape = shape,
            initializer = tf.initializers.he_normal())
    
    def _shape_initialize(self, x_data_batch, y_data_batch) :
        self.input_shape = x_data_batch.shape
        self.output_shape = y_data_batch.shape

    def _data_load(self, x_data, embedding) :
        self.dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        self.dataset = self.dataset.shuffle(777).repeat().batch(self.batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
    
    def fit(self, x_data, y_data) :
        if self.dataset :
            pass
        else : 
            self._data_load(x_data, y_data)
        
        with tf.Session() as sess :
            sess.run(self.iterator.initializer)

    def slicer(self, word) :
        return None
    
    def lstm(self, x_data, y_data, session) :
        cell = tf.nn.rnn_cell.LSTMCell(256)
        outputs, state = tf.nn.dynamic_rnn(cell = cell,
                            inputs = x_data,
                            time_major = False,
                            dtype = tf.float32)
                            
        layer_name = 'dense_layer'
        with tf.variable_scope(layer_name, reuse = tf.AUTO_REUSE) :
            w_f = self._weight_variable(layer_name, [256, ])
            b_f = self._bias_variable(layer_name, )

        if self.is_training :
            outputs