"""
This file is used to train a word2vec model from training data
"""

# coding: utf-8

from gensim.models import word2vec
import os
from tqdm import tqdm
import argparse


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in tqdm(os.listdir(self.dirname)):
            if int(fname.split('.')[0]) <= last_id_in_train:
                for line in open(os.path.join(self.dirname, fname)):
                    yield line.split()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()
    
    # find the last bug id in training
    training_file = '../../SABD/dataset/{}/training_{}.txt'.format(args.project, args.project)
    with open(training_file, 'r') as f:
        lines = f.readlines()
    last_id_in_train = int(lines[1].split()[-1])

    PROJECT_PATH = '../data/preprocess/{}/'.format(args.project)
    os.makedirs(PROJECT_PATH, exist_ok=True)

    # Corpus contains bug reports in the training set
    sentences = MySentences(PROJECT_PATH)

    model = word2vec.Word2Vec(sentences, vector_size = 20, window = 5, min_count = 2, workers = 10, sg = 0)

    os.makedirs('../model/word2vec/', exist_ok=True)
    model.save('../model/word2vec/{}.model'.format(args.project))