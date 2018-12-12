import tensorflow as tf
import numpy as np
import os, sys
import time
from tqdm import tqdm, trange
import pickle
from argparse import ArgumentParser
import math

class LanguageModel() :
    def __init__(self, num_units = 256, epochs = 10, learning_rate = 0.001, embed_dim = 256, batch_size = 256) :
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_units = num_units
        self.embed_dim = embed_dim
        self.batch_size = batch_size

    def load_dataset(self, train_data_path, test_data_path, word_vocab_path) :
        with open(train_data_path, 'rb') as f :
            train_data, train_seq_length = pickle.load(f)
        
        with open(test_data_path, 'rb') as f :
            test_data, test_seq_length = pickle.load(f)
        
        with open(word_vocab_path, 'rb') as f :
            word_vocab = pickle.load(f)
            self.word_vocab = dict(word_vocab)
        self.train_step = int(len(train_data) / self.batch_size)
        self.vocab_size = len(self.word_vocab) - 1
        self.train_data = tf.data.Dataset.from_tensor_slices((train_data, train_seq_length)).shuffle(buffer_size = 100).batch(self.batch_size)
        self.iterator = self.train_data.make_initializable_iterator()
        self.test_data = tf.data.Dataset.from_tensor_slices((test_data, test_seq_length)).repeat()
        self.word_embeddings = tf.get_variable('embedding', [self.vocab_size, self.num_units], dtype = tf.float32)
        print('data loaded.')

    def _LSTMLayer(self, batch_x, seq_len) :
        embedding_selected = tf.nn.embedding_lookup(self.word_embeddings, batch_x)
        cell = tf.nn.rnn_cell.LSTMCell(num_units = self.embed_dim)
        self._initial_state = cell.zero_state(self.batch_size, tf.float32)
        start_tokens = tf.constant([[0, 0], [0, 1]], dtype = tf.int32)
        if self._is_train :
            batch_y = tf.pad(batch_x, start_tokens, mode = 'CONSTANT', constant_values = 1)
            batch_y = tf.slice(batch_y, [0, 1], [-1, 30])
        # outputs = []
        # state = self._initial_state
        # with tf.variable_scope('RNN') :
        #     for time_step in range(30) :
        #         if time_step > 0 :
        #             tf.get_variable_scope().reuse_variables()
        #         output, state = tf.nn.dynamic_rnn(cell = cell,
        #                                 inputs = embedding_selected[:, time_step:time_step + 1, :],
        #                                 initial_state = state,
        #                                 dtype = tf.float32)
        #         outputs.append(output)
        
        # output = tf.reshape(tf.stack(axis = 1, values = outputs), [-1, self.embed_dim])

        # softmax_w = tf.get_variable(
        #     "softmax_w", [self.embed_dim, self.vocab_size], dtype = tf.float32)
        # softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype = tf.float32)
        # logits = tf.matmul(output, softmax_w) + softmax_b
        # logits = tf.reshape(logits, [self.batch_size, 30, self.vocab_size])
        
        # return logits, batch_y
        with tf.variable_scope('RNN') :
            outputs, state = tf.nn.dynamic_rnn(cell = cell,
                                            inputs = embedding_selected,
                                            initial_state = self._initial_state,
                                            sequence_length = seq_len,
                                            dtype = tf.float32
                                            )

        output = tf.reshape(outputs, [-1, self.embed_dim])
        softmax_w = tf.get_variable(
            "softmax_w", [self.embed_dim, self.vocab_size], dtype = tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype = tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        logits = tf.reshape(logits, [self.batch_size, 30, self.vocab_size])
        return logits, batch_y

    def train(self, session, model_dir) :
        self._is_train = True

        best_saver = tf.train.Saver(max_to_keep = 1)
        last_saver = tf.train.Saver(max_to_keep = 1)
        loss = 0
        batch_x, seq_len = self.iterator.get_next()
        logits, batch_y = self._LSTMLayer(batch_x, seq_len)
        seq_loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            batch_y,
            tf.ones([self.batch_size, 30], dtype = tf.float32),
            average_across_batch = True
        )
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, self.train_step, decay_rate = 0.95)
        train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(seq_loss)
        session.run(tf.global_variables_initializer())
        self._best_loss = 99999
        loss_epoch = 0
        for epoch in trange(self.epochs) :
            session.run(self.iterator.initializer)
            for step in trange(self.train_step) :
                _, loss = session.run([train_op, seq_loss])
                loss_epoch += loss
            mean_loss = loss_epoch / self.train_step
            print('loss for epoch {} is {}. current best loss is {}.'.format(epoch, mean_loss, self._best_loss))
            loss_epoch = 0
            if mean_loss < self._best_loss :
                best_saver.save(session, os.path.join(model_dir, 'best', 'best_model'))
                self._best_loss = mean_loss
        last_saver.save(session, os.path.join(model_dir, 'last', 'last_model'))
    
    def _word_indexer(self, text) :
        words = text.split()
        words_index = np.array([self.word_vocab[word] for word in words if word in self.word_vocab])
        text_length = np.array([len(words_index)])
        words_index = np.pad(words_index, (0, 30 - len(words_index)), 'constant', constant_values = 1).reshape(1, 30, 1)
        return words_index, text_length

    def get_tensor(self, session, text) :
        self._is_train = False
        input, seq_len = self._word_indexer(text)
        output = self._LSTMLayer(input, seq_len)
        return session.run(output)

    def get_embeddings(self, session) :
        return self.word_embeddings.eval(session)

    def load_graph(self, session, path) :
        saver = tf.train.Saver()
        saver.restore(session, path)

def get_parser() :
    parser = ArgumentParser('model_spec')
    parser.add_argument('--num_units', '-n', default = 256, type = int)
    parser.add_argument('--epochs', '-e', default = 10, type = int)
    parser.add_argument('--learning_rate', '-l', default = 0.001, type = float)
    parser.add_argument('--dimension', '-d', default = 256, type = int)
    parser.add_argument('--batch_size', '-b', default = 256, type = int)
    return parser

def main() :
    parser = get_parser()
    args = parser.parse_args()

    tester = LanguageModel(num_units = args.num_units, epochs = args.epochs, embed_dim = args.dimension, batch_size = args.batch_size)
    tester.load_dataset('../data/wikinews/wiki_train.pkl', '../data/wikinews/wiki_test.pkl', '../data/wikinews/word_vocab.pkl')
    with tf.Session() as sess :
        tester.train(sess, '../model/wikinews/Language_model')
    
if __name__ == '__main__' :
    main()