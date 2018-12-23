import tensorflow as tf
import numpy as np
import os, sys
import time
from tqdm import tqdm, trange
import pickle
from argparse import ArgumentParser
import math

class Word_Predictor() :
    def __init__(self, epochs = 100, learning_rate = 0.001, batch_size = 256) : 
        self._epochs = epochs
        self._start_learning_rate = learning_rate
        self._batch_size = batch_size
    
    def data_load_initialize(self, train_data_path, test_data_path, word_vocab_path) :
        with open(train_data_path, 'rb') as f :
            train_data, train_seq_length = pickle.load(f)
        
        with open(test_data_path, 'rb') as f :
            test_data, test_seq_length = pickle.load(f)
        
        with open(word_vocab_path, 'rb') as f :
            word_vocab = pickle.load(f)
            word_vocab = dict(word_vocab)
            self.reverse_dict = {v : k for k, v in word_vocab.items()}
        
        self._train_step = int(len(train_data) / self._batch_size)
        self._train_data = tf.data.Dataset.from_tensor_slices((train_data, train_seq_length)).shuffle(buffer_size = 100).batch(self._batch_size)
        self._iterator = self._train_data.make_initializable_iterator()

    def _make_croped_data(self, word) : 
        


if __name__ == '__main__' :
    temp = Word_Predictor()
    temp.data_load_initialize('../data/wikinews/wiki_train.pkl', '../data/wikinews/wiki_test.pkl', '../data/wikinews/word_vocab.pkl')