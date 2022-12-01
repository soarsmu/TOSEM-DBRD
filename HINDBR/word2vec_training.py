"""
Created on 5 Aug 2021
"""

import logging
from gensim.models import Word2Vec
from datetime import datetime
from utils import get_logger
import codecs
import ujson
import argparse
import os

class bug_corpus:
    def __init__(self, json_lines):
        self.json_lines = json_lines

    def __iter__(self):
        for line in self.json_lines:
            cur_dict = ujson.loads(line)
            document = cur_dict['short_desc'] + cur_dict['description']

            if document != None:
                yield line.split()


def train_word2vec():
    logging.info("This may take a while...")
    start_time = datetime.now()
    logging.info('It started at: %s' % start_time)

    logging.info("reading the json file: {0}".format(PROJECT))
    with codecs.open(JSON_FILE_PATH, 'r', 'utf-8') as f:
        lines = f.readlines()
    
    sentences = bug_corpus(lines)
    bug_w2c_model = Word2Vec(
        sentences, 
        #  size=100, # Dimensionality 
        vector_size=100,
        window=10,# 5 for cbow, 10 for sg
        min_count=5,
        workers=10,
        sg=1 # 0 for cbow, 1 for sg
    )
    
    os.makedirs('data/pretrained_embeddings/word2vec', exist_ok=True)
    bug_w2c_model.wv.save_word2vec_format(  \
        'data/pretrained_embeddings/word2vec/{}-vectors-gensim-sg100dwin10.bin'.format(PROJECT), binary=True)

    end_time = datetime.now()
    logging.info('Completed after: {}'.format(end_time - start_time))
    logging.info("Training is done and the model is saved!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name')
    
    args = parser.parse_args()
    PROJECT = args.project
    
    # use the cleaned json, no more cleaning needed
    JSON_FILE_PATH = '../SABD/dataset/{}/{}_soft_clean.json'.format(PROJECT, PROJECT)
    os.makedirs('./log', exist_ok=True)
    get_logger('./log/word2vec_training_{}_{}.log'.format(PROJECT, \
        datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

    train_word2vec()