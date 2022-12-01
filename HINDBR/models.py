"""
Train the hin_text model
"""

import sys
sys.path.append('./')
from time import time
import pandas as pd
pd.set_option("max_columns", None)
from tensorflow import keras

from imblearn.combine import SMOTETomek

from modules import convert_train_valid_to_ids, fit_tokenizer,  \
    prepare_word_embedding_matrix, prepare_hin_embedding_matrix,  \
        get_max_sequence, save_corpus_ids

import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping
import datetime
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Embedding, \
    Lambda, concatenate, Dropout, Bidirectional, LSTM
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adadelta
import os
from utils import get_logger
import logging
import argparse

# MODEL_NO = 5
# 4: LR=0.1
# 3: LR=1.0
# 5: changed the valid data

TEXT_DATA = 'summary_'

# Word embedding vector
WORD_EMBEDDING_DIM = '100'
EMBEDDING_ALGO = 'sg'
HIN_EMBEDDING_DIM = '128'

os.makedirs('./output/trained_model/', exist_ok=True)
os.makedirs('./output/training_history/', exist_ok=True)
os.makedirs('./data/model_training/', exist_ok=True)
os.makedirs('log/', exist_ok=True)

my_path = os.path.dirname(os.path.abspath(__file__))


tf.compat.v1.random.set_random_seed(941207)

def train_text():
    start_time = datetime.datetime.now()
    logging.info('It started at: %s' % start_time)

    max_seq_length = get_max_sequence(PROJECT)
    logging.info('max seq length is {}'.format(max_seq_length))
    
    corpus_pkl = 'data/model_training/{}_corpus.pkl'.format(PROJECT)
    
    ### fit tokenizer
    cur_tokenizer = fit_tokenizer(corpus_pkl)
    embedding_matrix = prepare_word_embedding_matrix(cur_tokenizer, PROJECT)
    
    if not os.path.exists('data/model_training/{}_corpus_ids.pkl'.format(PROJECT)):
        logging.info('generating corpus ids file')
        save_corpus_ids(corpus_pkl, cur_tokenizer, max_seq_length, PROJECT)
    
    training_pair_pkl = 'data/model_training/{}_train_complete.pkl'.format(PROJECT)
    
    if not os.path.exists(training_pair_pkl):
        logging.info('generating training pairs pkl...')
        convert_train_valid_to_ids(training_pair_ids, PROJECT, training_pair_pkl)
        
    train_pair_df = pd.read_pickle(training_pair_pkl)
    
    # Model variables
    n_hidden_rnn = 100
    gradient_clipping_norm = 1.25
    batch_size = 128
    n_epoch = 100
    
    #####################
    ### I: Model Text (Text) ###
    K.clear_session()
    #####################
    
    MODEL_NAME = 'TEXT'
    logging.info('training {} model'.format(MODEL_NAME))
    K.clear_session()
    #################################

    ## Text Information Representation ##
    # 1) Text Input Layer
    bug_text_left_input = Input(shape=(max_seq_length,), dtype='int32', name='text_left_input')
    bug_text_right_input = Input(shape=(max_seq_length,), dtype='int32', name='text_right_input')

    # 2) Embedding Layer
    num_tokens = len(cur_tokenizer.word_index) + 2
    embedding_layer = Embedding(
        input_dim = num_tokens, 
        output_dim = int(WORD_EMBEDDING_DIM),
        weights = [embedding_matrix], 
        # embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        input_length = max_seq_length, 
        name = 'text_embedding'
    )

    bug_text_embedding_left = embedding_layer(bug_text_left_input)
    bug_text_embedding_right = embedding_layer(bug_text_right_input)

    # 3) Shared Bi-LSTM 
    shared_bilstm = Bidirectional(
        LSTM(n_hidden_rnn, return_sequences=False, name='shared_bilstm')
    )
    bug_text_left_bilstm = shared_bilstm(bug_text_embedding_left)
    bug_text_right_bilstm = shared_bilstm(bug_text_embedding_right)
    bug_text_left_repr = Dropout(0.25)(bug_text_left_bilstm)
    bug_text_right_repr = Dropout(0.25)(bug_text_right_bilstm)


    ## Malstm Distance Layer ##
    mahanttan_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
    mahanttan_distance_text = mahanttan_layer([bug_text_left_repr, bug_text_right_repr])

    ## Build the model ##
    model_text = Model(inputs=[bug_text_left_input, bug_text_right_input], \
        outputs=[mahanttan_distance_text])
    optimizer = Adadelta(clipnorm=gradient_clipping_norm, learning_rate=1.0)
    model_text.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    model_text.summary()
    
    # Split to dicts -- train
    X_text_train = {
        'left': np.array([item for item in train_pair_df['text_left']]), \
        'right': np.array([item for item in train_pair_df['text_right']])
    }
    
    # Convert labels to their numpy representations
    Y_train = train_pair_df['is_duplicate'].values
    Y_train = Y_train.astype('int')
    
    X_train_cat = np.concatenate([X_text_train['left'], X_text_train['right']], axis=-1)
    # print(X_text_train['left'].shape)
    text_features_num = np.shape(X_text_train['left'])[1]
    
    smt = SMOTETomek(random_state=42,n_jobs=14)
    X_train_cat_res, Y_train_res = smt.fit_resample(X_train_cat, Y_train)
    X_text_train_left_res = X_train_cat_res[:, 0 : text_features_num]
    X_text_train_right_res = X_train_cat_res[:, 1 * text_features_num:]
    
    ## Train the model
    training_start_time = time()
    model_text.fit(x = [X_text_train_left_res, X_text_train_right_res], 
        y = Y_train_res, 
        batch_size=batch_size, 
        epochs=n_epoch,
        validation_split=0.2, 
        shuffle=True,
        # callbacks = [early_stopping_monitor]
    )
    
    logging.info("Training time finished.\n{} epochs in {}".format(\
        n_epoch, datetime.timedelta(seconds=time()-training_start_time)))
    
    ## Save model ## 
    for i in range(1, 11):
        if not os.path.exists(MODEL_SAVE_FILE + MODEL_NAME + '_{}.h5'.format(i)):
            model_save_name = MODEL_SAVE_FILE + MODEL_NAME + '_{}.h5'.format(i)
            break
    model_text.save(model_save_name)

    del model_text
    # del model_trained
    K.clear_session()


def train_text_hin():
    start_time = datetime.datetime.now()
    logging.info('It started at: %s' % start_time)

    max_seq_length = get_max_sequence(PROJECT)
    logging.info('max seq length is {}'.format(max_seq_length))
    
    corpus_pkl = 'data/model_training/{}_corpus.pkl'.format(PROJECT)
    
    ### fit tokenizer
    cur_tokenizer = fit_tokenizer(corpus_pkl)
    cur_word_embeddings = prepare_word_embedding_matrix(cur_tokenizer, PROJECT)
    cur_hin_embeddings = prepare_hin_embedding_matrix(corpus_pkl, PROJECT)
    
    if not os.path.exists('data/model_training/{}_corpus_ids.pkl'.format(PROJECT)):
        logging.info('generating corpus ids file')
        save_corpus_ids(corpus_pkl, cur_tokenizer, max_seq_length, PROJECT)
    
    training_pair_pkl = 'data/model_training/{}_train.pkl'.format(PROJECT)
    valid_pair_pkl = 'data/model_training/{}_valid.pkl'.format(PROJECT)
    
    if not os.path.exists(training_pair_pkl):
        logging.info('generating training pairs pkl...')
        convert_train_valid_to_ids(training_pair_ids, PROJECT, True)
        
    train_pair_df = pd.read_pickle(training_pair_pkl)
    valid_pair_df = pd.read_pickle(valid_pair_pkl)
    
    # Model variables
    n_hidden_rnn = 100
    n_dense_hin = 32
    n_dense_fusion = 64
    gradient_clipping_norm = 1.25
    batch_size = 128
    n_epoch = 100
    
    # early_stopping_monitor = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0,
    #     patience=0,
    #     verbose=0,
    #     mode='auto',
    #     baseline=None,
    #     restore_best_weights=True
    # )
    #####################
    ### I: Model Text (Text) ###
    K.clear_session()
    #####################
    
    MODEL_NAME = 'TEXT_HIN_DENSE'
    logging.info('training {} model'.format(MODEL_NAME))
    K.clear_session()
    #################################

    ## Text Information Representation ##
    # 1) Text Input Layer
    bug_text_left_input = Input(shape=(max_seq_length,), dtype='int32', name='text_left_input')
    bug_text_right_input = Input(shape=(max_seq_length,), dtype='int32', name='text_right_input')

    # 2) Embedding Layer
    embedding_layer = Embedding(
        input_dim = len(cur_tokenizer.word_index) + 1, 
        output_dim = int(WORD_EMBEDDING_DIM), 
        weights = [cur_word_embeddings], 
        input_length = max_seq_length, 
        trainable = False,
        name = 'text_embedding'
    )

    bug_text_embedding_left = embedding_layer(bug_text_left_input)
    bug_text_embedding_right = embedding_layer(bug_text_right_input)

    # 3) Shared Bi-LSTM 
    shared_bilstm = Bidirectional(
        LSTM(n_hidden_rnn, return_sequences=False, name='shared_bilstm')
    )
    bug_text_left_bilstm = shared_bilstm(bug_text_embedding_left)
    bug_text_right_bilstm = shared_bilstm(bug_text_embedding_right)
    bug_text_left_repr = Dropout(0.25)(bug_text_left_bilstm)
    bug_text_right_repr = Dropout(0.25)(bug_text_right_bilstm)


    ## Hin Information Representation ##
    # 1) Hin Input Layer
    bug_hin_left_input = Input(shape=(6,), dtype='int32', name='hin_left_input')
    bug_hin_right_input = Input(shape=(6,), dtype='int32', name='hin_right_input')

    # 2) Embedding Layer
    embedding_layer = Embedding(
        input_dim = len(cur_hin_embeddings),
        output_dim = int(HIN_EMBEDDING_DIM),
        weights = [cur_hin_embeddings],
        input_length = 6,
        trainable = False,
        name = 'hin_embedding'
    )

    bug_hin_embedding_left = embedding_layer(bug_hin_left_input)
    bug_hin_embedding_right = embedding_layer(bug_hin_right_input)
    bug_hin_embedding_left_flat = Flatten()(bug_hin_embedding_left)
    bug_hin_embedding_right_flat = Flatten()(bug_hin_embedding_right)
    dense_layer = Dense(n_dense_hin, activation='tanh')
    bug_hin_left_repr = dense_layer(bug_hin_embedding_left_flat)
    bug_hin_right_repr = dense_layer(bug_hin_embedding_right_flat)

    ## Bug Report Representation ##
    merge_bug_text_hin_left = concatenate([bug_text_left_repr, bug_hin_left_repr])
    merge_bug_text_hin_right = concatenate([bug_text_right_repr, bug_hin_right_repr])
    dense_layer = Dense(n_dense_fusion, activation='tanh', name='dense_bugrepr')
    bug_left_repr = dense_layer(merge_bug_text_hin_left)
    bug_right_repr = dense_layer(merge_bug_text_hin_right)

    ## Malstm Distance Layer ##
    mahanttan_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
    mahanttan_distance_text_hin_dense = mahanttan_layer([bug_left_repr, bug_right_repr])
    
    ## Build the model ##
    model_text_hin_dense = Model(
        inputs=[bug_text_left_input, 
                bug_text_right_input, 
                bug_hin_left_input, 
                bug_hin_right_input], 
        outputs=[mahanttan_distance_text_hin_dense]
    )
    optimizer = Adadelta(clipnorm=gradient_clipping_norm, learning_rate=1.0)
    model_text_hin_dense.compile(loss='binary_crossentropy', \
        optimizer=optimizer, metrics=['accuracy'])
    model_text_hin_dense.summary()


    ## Train the model
    training_start_time = time()

    model_text_hin_dense.fit(x = [
        np.array([item for item in train_pair_df['text_left']]), 
        np.array([item for item in train_pair_df['text_right']]), 
        np.array([item for item in train_pair_df['hin_left']]), 
        np.array([item for item in train_pair_df['hin_right']])
        ], y = train_pair_df['is_duplicate'].values, 
        batch_size=batch_size, 
        epochs=n_epoch,
        validation_data=([
            np.array([item for item in valid_pair_df['text_left']]),
            np.array([item for item in valid_pair_df['text_right']]),
            np.array([item for item in valid_pair_df['hin_left']]),
            np.array([item for item in valid_pair_df['hin_right']])
        ], valid_pair_df['is_duplicate'].values),
        shuffle=True,
        # callbacks = [early_stopping_monitor]
    )
    
    logging.info("Training time finished.\n{} epochs in {}".format(n_epoch, \
        datetime.timedelta(seconds=time()-training_start_time)))
    
    ## Save model ## 
    for i in range(1, 11):
        if not os.path.exists(MODEL_SAVE_FILE + MODEL_NAME + '_{}.h5'.format(i)):
            model_save_name = MODEL_SAVE_FILE + MODEL_NAME + '_{}.h5'.format(i)
            break
    model_text_hin_dense.save(model_save_name)

    del model_text_hin_dense
    # del model_trained
    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name', required=True)
    parser.add_argument('--variant', help='model variant', required=True)
    
    args = parser.parse_args()
    
    PROJECT = args.project
    VARIANT = args.variant
    
    training_pair_ids = 'data/model_training/{}_training_pairs.txt'.format(PROJECT)
    ### test the word embeddings
    
    # Model Save    
    MODEL_SAVE_FILE = 'output/trained_model/' + PROJECT + '_' + EMBEDDING_ALGO + \
        WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA

    # Model Training history record
    EXP_HISTORY_ACC_SAVE_FILE = 'output/training_history/' + 'acc_' + PROJECT + \
        '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    EXP_HISTORY_VAL_ACC_SAVE_FILE = 'output/training_history/' + 'val_acc_'+ PROJECT + \
        '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    EXP_HISTORY_LOSS_SAVE_FILE = 'output/training_history/' + 'loss_' + PROJECT + '_' + \
        EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    EXP_HISTORY_VAL_LOSS_SAVE_FILE = 'output/training_history/' + 'val_loss_' + PROJECT + \
        '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 

    # Model Test history record
    EXP_TEST_HISTORY_FILE = 'output/training_history/' + 'test_result_' + PROJECT + '_' + \
        EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 

    get_logger('./log/train_model_{}_{}_{}.log'.format(PROJECT, VARIANT, \
        datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

    if VARIANT == 'text':
        train_text()
    elif VARIANT:
        train_text_hin()