import os, sys
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--language', help = 'select language from eng, e, kor, or k')
parser.add_argument('-m', '--model', help = 'select embedding model from fasttext, f, word2vec, w, elmo, or e')
parser.add_argument('-d', '--dataset', help = 'select dataset from \n [ENG] \n twitter, t, sms, s, keyboard, k \n [KOR] \n movie, m, kin, k')
parser.add_argument('-s', '--size', help = 'embedding size', default = 300)
parser.add_argument('-e', '--epochs', help = 'total epochs', default = 10)
parser.add_argument('-r', '--learning_rate', help = 'learning_rate for training', default = 0.025)

arguments = parser.parse_args()
embed_model = utils.Embedding_Model(arguments)
embed_model.train_model()
embed_model.save_model()
