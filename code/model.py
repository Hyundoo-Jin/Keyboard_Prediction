import tensorflow as tf
import numpy as np
import os, sys
import time
from tqdm import tqdm
import time
import pickle

class LanguageModel() :
    def __init__(self, num_units = 256, epochs = 300, learning_rate = 0.001) :
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_units = num_units

    def _modelgraph(self, inputs, outputs) :
        with tf.variable_scope('model') :
            train_x, train_y = inputs, outputs
        input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_x, 1)), 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_y, 1)), 1)
        cell = tf.contrib.rnn.LSTMCell(self.num_units)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell, train_x, dtype = tf.float32)
        train_helper = tf.contrib.seq2seq.TrainingHelper(train_y)
