"""
Use the learned word2vec model to represent each bug report
"""

from gensim.models import word2vec
import numpy as np
import os
import pickle
from tqdm import tqdm
import argparse
import json


def toMatrix(model):
    txt_files = os.listdir(txt_path)

    for file in tqdm(txt_files): 
        # contains preprocessed bug reports
        # modified the vector size
        matrix = np.zeros((300, 20))
        count = 0

        with open(os.path.join(txt_path, file)) as f:
            for line in f.readlines():
                words = line.split()
                for word in words:
                    if count < 300:
                        if word in model.wv.index_to_key:
                            matrix[count,:] = model.wv[word]
                            count += 1
                        else:
                            count += 1
        
        with open(matrix_path + file.split('.')[0]+'.pkl', 'wb') as output:
            pickle.dump(matrix, output)


def toFullMatrix():
    sabd_data_path = '../../SABD/dataset/{}/'.format(args.project)
    with open(sabd_data_path + '{}_soft_clean.json'.format(args.project), 'r') as f:
        data = f.readlines()

    full_matrix = np.zeros((len(data), 300, 20, 1))
    i = 0

    for data_line in tqdm(data):
        br = json.loads(data_line)

        with open(matrix_path + '{}.pkl'.format(br['bug_id']), 'rb') as input:
            matrix = pickle.load(input).reshape(300, 20, 1)
            full_matrix[i] = matrix
            i += 1

    with open(matrix_path + 'full_matrix.npy', 'wb') as f:
        np.save(f, full_matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name')

    args = parser.parse_args()

    txt_path = '../data/preprocess/{}/'.format(args.project)
    os.makedirs(txt_path, exist_ok=True)

    matrix_path = '../data/matrix/{}/br/'.format(args.project)
    os.makedirs(matrix_path, exist_ok=True)

    model_path = '../model/word2vec/{}.model'.format(args.project)

    model = word2vec.Word2Vec.load(model_path)
    # toMatrix(model)
    toFullMatrix()