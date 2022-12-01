"""
Created on 20 July 2021
"""

import sys
sys.path.append('./')

from utils import get_logger
from datetime import datetime
import logging
import time
import math
import numpy as np
from tensorflow import keras
from modules import fit_tokenizer, get_max_sequence
from modules import BugReportDatabase, BugDataset, SunRanking
from tqdm import tqdm
import argparse
import pickle
import os

TEXT_DATA = 'summary_'

# Word embedding vector
WORD_EMBEDDING_DIM = '100'
EMBEDDING_ALGO = 'sg'

class GeneralScorerText(object):

    def __init__(self, model):
        self.model = model

    def score(self, bug_id, candidate_list):
        text_left, text_right = list(), list()
        
        for cand in candidate_list:
            text_left.append(bid_representations[bug_id]['summary'])
            text_right.append(bid_representations[cand]['summary'])
        
        result = self.model.predict([
                np.array(text_left), 
                np.array(text_right)
            ], batch_size=128
        )
        # result = self.model.predict(np.array(data_test))
        preds_test = np.array([x[0] for x in result])
        return preds_test


class GeneralScorerTextHinDense(object):
    def __init__(self, model):
        self.model = model

    def score(self, bug_id, candidate_list):
        text_left, text_right = list(), list()
        hin_left, hin_right = list(), list()
        
        for cand in candidate_list:
            text_left.append(bid_representations[bug_id]['summary'])
            text_right.append(bid_representations[cand]['summary'])
            hin_left.append(bid_representations[bug_id]['hin'])
            hin_right.append(bid_representations[cand]['hin'])
        
        result = self.model.predict([
                np.array(text_left), 
                np.array(text_right),
                np.array(hin_left),
                np.array(hin_right)
            ],
            batch_size=128
        )

        # result = self.model.predict(np.array(data_test))
        preds_test = np.array([x[0] for x in result])

        return preds_test


def generateRecommendationList(cur_bug_id, candidates, scorer):
    similarityScores = scorer.score(cur_bug_id, candidates)
    bugScores = [(bugId, score) for bugId, score in zip(candidates, similarityScores) if bugId != cur_bug_id]
    # Sort in descending order the bugs by probability of being duplicate
    sortedBySimilarity = sorted(bugScores, key=lambda x: x[1], reverse=True)

    return sortedBySimilarity


class RecallRate(object):
    def __init__(self, bugReportDatabase, k = None):

        self.masterSetById = bugReportDatabase.getMasterSetById()
        self.masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

        if k is None:
            k = list(range(1, 21))

        self.k = sorted(k)

        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0
        self.logger = logging.getLogger()

    def reset(self):
        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0

    def update(self, anchorId, recommendationList):
        mastersetId = self.masterIdByBugId[anchorId]
        masterSet = self.masterSetById[mastersetId]
        # biggestKValue = self.k[-1]

        # pos = biggestKValue + 1
        pos = math.inf
        correct_cand = None

        if len(recommendationList) == 0:
            self.logger.warning("Recommendation list of {} is empty. \
                Consider it as miss.".format(anchorId))
        else:
            seenMasters = set()

            for bugId, p in recommendationList:
                mastersetId = self.masterIdByBugId[bugId]

                if mastersetId in seenMasters:
                    continue

                seenMasters.add(mastersetId)

                if bugId in masterSet:
                    pos = len(seenMasters)
                    correct_cand = bugId
                    break

        # If one of k duplicate bugs is in the list of duplicates, 
        # so we count as hit. We calculate the hit for each different k
        for idx, k in enumerate(self.k):
            if k < pos:
                continue

            self.hitsPerK[k] += 1

        self.nDuplicate += 1

        return pos, correct_cand

    def compute(self):
        recallRate = {}
        for k, hit in self.hitsPerK.items():
            rate = float(hit) / self.nDuplicate
            recallRate[k] = rate
        return recallRate


# position and recall rate are given here
def logRankingResult(logger, duplicateBugs, rankingScorer, bugReportDatabase, recommendationListfn = generateRecommendationList):
    rankingClass = SunRanking(bugReportDatabase, recallRateDataset, 365)

    recallRateMetric = RecallRate(bugReportDatabase)
    start_time = time.time()
    positions = []

    for i, duplicateBugId in enumerate(tqdm(duplicateBugs)):
        candidates = rankingClass.getCandidateList(duplicateBugId)

        if i > 0 and i % 500 == 0:
            logger.info('RR calculation - {} duplicate reports were processed'.format(i))

        if len(candidates) == 0:
            logging.getLogger().warning("Bug {} has 0 candidates!".format(duplicateBugId))
            recommendation = list()
        else:
            recommendation = recommendationListfn(duplicateBugId, candidates, rankingScorer)

        # Update the metrics
        pos, correct_cand = recallRateMetric.update(duplicateBugId, recommendation)

        positions.append(pos)


    recallRateResult = recallRateMetric.compute()

    end_time = time.time()
    nDupBugs = len(duplicateBugs)
    duration = end_time - start_time

    logger.info(
        '[Recall Rate] Throughput: {} bugs per second (bugs={} ,seconds={})' \
        .format( (nDupBugs / duration), nDupBugs, duration)
    )

    for k, rate in recallRateResult.items():
        hit = recallRateMetric.hitsPerK[k]
        total = recallRateMetric.nDuplicate
        logger.info({
                'type': "metric", 
                'label': 'recall_rate', 
                'k': k, 
                'rate': rate, 
                'hit': hit,
                'total': total,
            })


    valid_queries = np.asarray(positions)
    MAP_sum = (1 / valid_queries).sum()
    MAP = MAP_sum / valid_queries.shape[0]

    logger.info({
            'type': "metric", 
            'label': 'MAP', 
            'value': MAP, 
            'sum': MAP_sum, 
            'total': valid_queries.shape[0],
        })

    logger.info('{}'.format(positions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name')
    parser.add_argument('--variant', help='model variant')
    parser.add_argument('--model_num', help='model num')
    
    args = parser.parse_args()
    
    PROJECT = args.project
    VARIANT = args.variant
    MODEL_NO = int(args.model_num)
    
    ### test the word embeddings
    WORD_EMBEDDING_FILE = 'data/pretrained_embeddings/word2vec/{}-vectors-gensim-'.format(PROJECT) + EMBEDDING_ALGO + WORD_EMBEDDING_DIM +'dwin10_1.bin'

    # HIN embedding vector
    HIN_EMBEDDING_DIM = '128'
    HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + PROJECT + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'
    HIN_NODE_DICT = 'data/hin_node_dict/' + PROJECT + '_node.dict'

    # Model Save
    MODEL_SAVE_FILE = 'output/trained_model/' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA

    # Model Training history record
    EXP_HISTORY_ACC_SAVE_FILE = 'output/training_history/' + 'acc_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    EXP_HISTORY_VAL_ACC_SAVE_FILE = 'output/training_history/' + 'val_acc_'+ PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    EXP_HISTORY_LOSS_SAVE_FILE = 'output/training_history/' + 'loss_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    EXP_HISTORY_VAL_LOSS_SAVE_FILE = 'output/training_history/' + 'val_loss_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 

    # Model Test history record
    EXP_TEST_HISTORY_FILE = 'output/training_history/' + 'test_result_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA 
    
    if VARIANT == 'text':
        MODEL_NAME = 'TEXT'
    else:
        MODEL_NAME = 'TEXT_HIN_DENSE'

    full_model_name = MODEL_SAVE_FILE + MODEL_NAME + '_' + str(MODEL_NO) + '.h5'
    
    os.makedirs('./result_log', exist_ok=True)
    logger = get_logger('./result_log/hindbr_{}_{}_{}.log'.format(PROJECT, MODEL_NAME, datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

    logger.info('Evaluating {} {} {}'.format(PROJECT, MODEL_NAME, MODEL_NO))
    start_time = datetime.now()

    logger.info('It started at: %s' % start_time)

    bug_report_database = BugReportDatabase.fromJson('../SABD/dataset/{}/{}_soft_clean.json'.format(PROJECT, PROJECT))
    recallRateDataset = BugDataset('../SABD/dataset/{}/test_{}.txt'.format(PROJECT, PROJECT))
    corpus_pkl = './data/model_training/{}_corpus.pkl'.format(PROJECT)
    
    ### fit tokenizer
    max_seq = get_max_sequence(PROJECT)
    
    cur_tokenizer = fit_tokenizer(corpus_pkl)
    duplicate_bugs = recallRateDataset.duplicateIds
    hindbr_model = keras.models.load_model(full_model_name)
    
    with open('data/model_training/{}_bid_corpus.pkl'.format(PROJECT), 'rb') as f:
        bid_representations = pickle.load(f)
        
    general_scorer = GeneralScorerTextHinDense(hindbr_model)

    logRankingResult(logger, duplicate_bugs, general_scorer, bug_report_database)
    
    end_time = datetime.now()
    logging.info('It completed at: {}'.format(end_time))
    logging.info('Completed after: {}'.format(end_time - start_time))