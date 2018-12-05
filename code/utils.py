import os, sys
from gensim.models import Word2Vec, FastText
import pickle
import time
import psutil
import glob

class Embedding_Model() :
    def __init__(self, arguments) :
        self.argument_setting(arguments)
        self.version = '{}_{}'.format(self.model, self.dataset)
        self.model_path = os.path.join('../model', self.dataset)
        self.dataset_path = os.path.join('../data', self.dataset)
        self.size = arguments.size
        self.epochs = arguments.epochs
        self.learning_rate = arguments.learning_rate

    def argument_setting(self, arguments) :
        # set model
        if arguments.model == 'f' or arguments.model == 'fasttext' :
            self.model = 'fasttext'
        elif arguments.model == 'w' or arguments.model == 'word2vec' :
            self.model = 'word2vec'
        elif arguments.model == 'e' or arguments.model == 'elmo' :
            self.model = 'elmo'
        else :
            raise AttributeError('Undefined model.')
        
        # set dataset
        if arguments.dataset == 'wikinews' or arguments.dataset == 'w' :
            self.dataset = 'wikinews'
        elif arguments.dataset == 'wordprediction' or arguments.dataset == 'p' :
            self.dataset = 'wordprediction'
        else :
            raise AttributeError('Undefined dataset.')
        
        print('Arguments seted. \n model : {} \n dataset : {}'.format(self.model, self.dataset))

    def get_version(self) :
        return self.version

    def train_model(self) :
        start_time = time.time()
        files = glob.glob(self.dataset_path + '/*.txt')
        temp_data = list()
        for file in files :
            with open(file, 'r') as f :
                temp_data.extend(f.readlines())
        print('data_loaded. Elapsed {} seconds'.format(round(time.time() - start_time, 3)))
        start_time = time.time()
        if self.model == 'fasttext' :
            self.model = FastText(list(map(lambda x : x.split(), temp_data)), size = self.size, iter = self.epochs, alpha = self.learning_rate, workers = psutil.cpu_count())
        elif self.model == 'word2vec' :
            self.model = Word2Vec(list(map(lambda x : x.split(), temp_data)), size = self.size, iter = self.epochs, alpha = self.learning_rate, workers = psutil.cpu_count())
        else :
            pass
        del temp_data # memory saving
        print('model fitted. Elapsed {} seconds'.format(round(time.time() - start_time, 3)))
    
    def save_model(self, file_path = None) :
        if file_path is None :
            file_path = self.model_path
        model_path = os.path.join(file_path, self.version + '.model') 
        self.model.save(model_path)
        print('Model saved. path : {}'.format(model_path))