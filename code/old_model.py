import tensorflow as tf
import numpy as np
import os, sys
import time
from tqdm import tqdm
import time
import pickle

class LanguageModel() :
    def __init__(self, num_units = 256, epochs = 300, learning_rate = 0.001, embed_dim = 256, batch_size = 128) :
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
            self.word_vocab = pickle.load(f)
        self.vocab_size = len(self.word_vocab) - 1
        self.train_data = tf.data.Dataset.from_tensor_slices((train_data, train_seq_length)).shuffle(buffer_size = 100).batch(self.batch_size)
        self.iterator = self.train_data.make_one_shot_iterator()
        self.test_data = tf.data.Dataset.from_tensor_slices((test_data, test_seq_length)).repeat()
        print('data loaded.')

    def _modelgraph(self, mode, features, labels, params) :
        inputs = outputs = features
        temp_length = tf.reshape(labels, [-1])
        input_length = output_length = temp_length
        input_embed = tf.contrib.layers.embed_sequence(
            inputs, vocab_size = self.vocab_size, embed_dim = self.embed_dim, scope = 'embed')
        output_embed = tf.contrib.layers.embed_sequence(
            outputs, vocab_size = self.vocab_size, embed_dim = self.embed_dim, scope = 'embed', reuse = True)
        with tf.variable_scope('embed', reuse = True):
            embeddings = tf.Variable('embeddings')

        start_token = tf.to_int32([3] * self.batch_size)

        encoder_cell = tf.contrib.rnn.GRUCell(num_units = self.num_units)
        
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell, inputs = input_embed, dtype = tf.float32)
        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_length)
        pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, strat_tokens = start_token, end_token = 4)

        def decode(helper, scope, reuse = None) :
            with tf.variable_scope(scope, reuse = reuse) :
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttension(
                    num_units = self.num_units, memory = encoder_outputs,
                    memory_sequence_length = input_length
                )
                decoder_cell = tf.contrib.rnn.LSTMCell(num_units = self.num_units)
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism, attention_layer_size = self.num_units / 2
                )
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    attention_cell, self.vocab_size, reuse = reuse
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = out_cell, helper = helper,
                    initial_state = out_cell.zero_state(
                        dtype = tf.float32, batch_size = self.batch_size
                    )
                )
                outputs = tf.contrib.seq2seq.dynamic_decode(
                    decoder = decoder, output_time_major = False,
                    impute_finished = True, maximum_iterations = 30
                )
                return outputs[0]

        train_outputs = decode(train_helper, 'decode')
        pred_outputs = decode(pred_helper, 'decode', reuse = True)

        tf.identity(train_outputs.sample_id[0], name = 'train_pred')
        weights = tf.to_float(tf.not_equal(train_outputs[:, : -1], 1))
        loss = tf.contrib.seq2seq.sequence_loss(
            train_outputs.rnn_output, outputs, weights = weights
        )
        train_op = tf.contrib.layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer = 'Adam',
            learning_rate = 0.001,
            summaries = ['loss', 'learning_rate']
        )

        tf.identity(pred_outputs.sample_id[0], name = 'predictions')
        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = pred_outputs.sample_id,
            loss = loss,
            train_op = train_op
        )
    
    def train(self, model_dir, params) :
        est = tf.estimator.Estimator(
            model_fn = self._modelgraph,
            model_dir = model_dir,
            params = params
            )
        
        print_inputs = tf.train.LoggingTensorHook(
            ['input_0', 'output_0'], every_n_iter = 100
        )
        print_predictions = tf.train.LoggingTensorHook(
            ['predictions', 'train_pred'], every_n_iter = 100
        )

        est.train(
            input_fn = self.iterator.get_next,
            hooks = [print_inputs, print_predictions],
            steps = 10000
        )


if __name__ == '__main__' :
    params = {
        'sequence_length' : 30
    }
    model = LanguageModel()
    model.load_dataset('../data/wikinews/wiki_train.pkl', '../data/wikinews/wiki_test.pkl', '../data/wikinews/word_vocab.pkl')
    model.train('../model', params)