"""
Created on 28 July 2022
"""

import sys
sys.path.append('./')
import argparse
from modules import BugReportDatabase, BugDataset, SunRanking
import logging
import gc
import time
from datetime import datetime
import math
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from utils import get_logger
import tensorflow as tf
import json
import numpy as np
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class GeneralScorer(object):
    def __init__(self, model):
        self.model = model
        with open(full_matrix_file, 'rb') as f:
            self.full_matrix = np.load(f)
            
        with open(sabd_data_path + '{}.json'.format(PROJECT)) as data:
            lines = data.readlines()

        self.bug_id_to_index = {}
        for i, line in zip(range(len(lines)), lines):
            cur_br = json.loads(line)
            self.bug_id_to_index[cur_br['bug_id']] = i

    def score(self, bug_id, candidates):
        cand_index = np.array([self.bug_id_to_index[cand] for cand in candidates])
        cand_matrix = self.full_matrix[cand_index]
        
        np.repeat(self.full_matrix[self.bug_id_to_index[bug_id]][None, :], \
            len(candidates), axis=0)
        
        test_data = np.concatenate((np.repeat(self.full_matrix[self.bug_id_to_index[bug_id]][None, :], \
            len(candidates), axis=0), cand_matrix), axis=3)
        
        result = self.model.predict(test_data, batch_size=128)
        
        del test_data
        del cand_index
        del cand_matrix
        gc.collect()

        # result的shape是(1,)
        preds_test = np.array([x[0] for x in result])

        del result
        gc.collect()
        return preds_test


def generateRecommendationList(cur_bug_id, candidates, scorer):
    similarityScores = scorer.score(cur_bug_id, candidates)

    # sort the candidates by their similarity scores
    bugScores = [(bugId, score) for bugId, score in zip(candidates, similarityScores) if bugId != cur_bug_id]
    # Sort in descending order the bugs by probability of being duplicate
    sortedBySimilarity = sorted(bugScores, key=lambda x: x[1], reverse=True)

    return sortedBySimilarity


class RecallRate(object):

    def __init__(self, bugReportDatabase, k=None, groupByMaster=True):

        self.masterSetById = bugReportDatabase.getMasterSetById()
        self.masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

        if k is None:
            k = list(range(1, 21))

        self.k = sorted(k)

        self.hitsPerK = dict((k, 0) for k in self.k)
        self.nDuplicate = 0
        self.logger = logging.getLogger()

        self.groupByMaster = groupByMaster

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
            self.logger.warning("Recommendation list of {} is empty. Consider it as miss.".format(anchorId))
        else:
            seenMasters = set()

            for bugId, p in recommendationList:
                mastersetId = self.masterIdByBugId[bugId]

                if self.groupByMaster:
                    if mastersetId in seenMasters:
                        continue

                    seenMasters.add(mastersetId)
                else:
                    seenMasters.add(bugId)

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
def logRankingResult(logger, duplicate_bugs, rankingScorer, bugReportDatabase, number, groupByMaster=True):
    rankingClass = SunRanking(bugReportDatabase, recallRateDataset, 365)
    recallRateMetric = RecallRate(bugReportDatabase, groupByMaster = groupByMaster)
    
    start_time = time.time()
    
    positions = []
    
    with open('ranking_result_{}_{}.txt'.format(PROJECT, number), 'w') as f:
        # cur_duplicate_bugs = duplicate_bugs[len(duplicate_bugs) // 4 * number:len(duplicate_bugs) // 4 * (number + 1)]
        index_list = list(np.array_split(range(len(duplicate_bugs)), 4)[number])
        cur_duplicate_bugs = duplicate_bugs[index_list[0] : index_list[-1] + 1]
        
        logging.info('bug index range is from {} to {}'.format(index_list[0], index_list[-1]))
        
        logging.info("the bug id range is {} to {}".format(cur_duplicate_bugs[0], cur_duplicate_bugs[-1]))
        
        for i, duplicateBugId in enumerate(tqdm(cur_duplicate_bugs)):
            candidates = rankingClass.getCandidateList(duplicateBugId)
            if i > 0 and i % 500 == 0:
                logger.info('RR calculation - {} duplicate reports were processed'.format(i))
            
            if len(candidates) == 0:
                logging.getLogger().warning("Bug {} has 0 candidates!".format(duplicateBugId))
                recommendation = list()
            else:
                recommendation = generateRecommendationList(duplicateBugId, candidates, rankingScorer)
            pos, correct_cand = recallRateMetric.update(duplicateBugId, recommendation)
            positions.append(pos)
            f.write(str(duplicateBugId) + ", " + str(pos))
            f.write('\n')
            
    recallRateResult = recallRateMetric.compute()

    end_time = time.time()
    nDupBugs = len(duplicate_bugs)
    duration = end_time - start_time
    logger.info(
        'Throughput: {} bugs per second (bugs={} ,seconds={})'\
            .format((nDupBugs / duration), nDupBugs, duration)
    )

    for k, rate in recallRateResult.items():
        hit = recallRateMetric.hitsPerK[k]
        total = recallRateMetric.nDuplicate
        logger.info({'type': "metric", 'label': 'recall_rate', 
                'k': k, 'rate': rate, 'hit': hit, 'total': total})
        
    valid_queries = np.asarray(positions)
    MAP_sum = (1 / valid_queries).sum()
    MAP = MAP_sum / valid_queries.shape[0]

    logger.info({'type': "metric", 'label': 'MAP', 'value': MAP,
    'sum': MAP_sum, 'total': valid_queries.shape[0]})

    logger.info('{}'.format(positions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enter project name.')
    parser.add_argument('--project', help='project name')
    parser.add_argument('--model_num', help='model num')
    parser.add_argument('--data_num', help='data num')
    
    args = parser.parse_args()
    
    PROJECT = args.project
    MODEL_NO = args.model_num
    DATA_NO = int(args.data_num)

    data_path = Path('../data/')
    model_path = Path('../model/{}'.format(PROJECT))

    dccnnModel = tf.keras.models.load_model(model_path / 'dccnn_{}.h5'.format(MODEL_NO))

    sabd_data_path = '../../SABD/dataset/{}/'.format(PROJECT)
    matrix_data_path = '../data/matrix/{}/'.format(PROJECT)
    full_matrix_file = matrix_data_path + 'br/full_matrix.npy'

    logger = get_logger('../result_log/dccnn_{}_{}_{}.log'\
        .format(PROJECT, DATA_NO, datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))
    
    start_time = datetime.now()
    logger.info('Evaluating {} {}'.format(PROJECT, MODEL_NO))
    logger.info('It started at: %s' % start_time)

    bug_report_database = BugReportDatabase.fromJson(sabd_data_path + '{}.json'.format(PROJECT))

    recallRateDataset = BugDataset(sabd_data_path + 'test_{}.txt'.format(PROJECT))

    duplicateBugs = recallRateDataset.duplicateIds
    general_scorer = GeneralScorer(dccnnModel)
    logging.info('loaded...')
    logRankingResult(logger, duplicateBugs, general_scorer, bug_report_database, DATA_NO)
    end_time = datetime.now()
    logging.info('It completed at: {}'.format(end_time))
    logging.info('Completed after: {}'.format(end_time - start_time))