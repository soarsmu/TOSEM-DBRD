"""
modified from https://github.com/bugzilla/bugzilla/blob/230d73a11989a46b0a0d3f271a1c4a260f371bd7/Bugzilla/Bug.pm
"""

import mysql.connector
import sys, os
sys.path.append('./')
import ujson
from tqdm import tqdm
from string import punctuation
import numpy as np
import math
import logging
import time
import re
import argparse
from modules import BugReportDatabase, BugDataset, SunRanking
from utils import get_logger

MAX_POSSIBLE_DUPLICATES = 25
FULLTEXT_OR = False

config = {
    'host': 'localhost',
    'port': 3306,
    # 'port': 33060,
    'user': 'root',
    'password': '12345678',
    'database': 'ITS',
}


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

        # pos = biggestKValue + 1
        pos = math.inf
        correct_cand = None

        if len(recommendationList) == 0:
            self.logger.warning("Recommendation list of {} is empty. Consider it as miss.".format(anchorId))
        else:
            seenMasters = set()

            for bugId in recommendationList:
                mastersetId = self.masterIdByBugId[bugId]

                if mastersetId in seenMasters:
                    continue

                seenMasters.add(mastersetId)

                if bugId in masterSet:
                    pos = len(seenMasters)
                    correct_cand = bugId
                    break

        # If one of k duplicate bugs is in the list of duplicates, so we count as hit. 
        # We calculate the hit for each different k
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


def get_bug_ids(predicted, candidate_content, candidates):
    """
    for all the recommendations, only keep those are real candidates with 365 days
    """

    recommended_bug_ids = list()
    recommended_bugs = list()
    
    real_candidates = set([candidate.strip() for candidate in candidates])

    for i, prediction in zip(range(len(predicted)), predicted):
        if str(prediction) in real_candidates:
            recommended_bug_ids.append(str(prediction))
            recommended_bugs.append(candidate_content[i])
            
    return recommended_bug_ids, recommended_bugs


def logRankingResult(duplicateBugs, recallRateDataset, bugReportDatabase, db_cursor):
    rankingClass = SunRanking(bugReportDatabase, recallRateDataset, 365)
    recallRateMetric = RecallRate(bugReportDatabase)
    start_time = time.time()
    positions = []
    
    predicted_summary = []
    
    count = 0
    for i, duplicateBugId in enumerate(tqdm(duplicateBugs)):
        candidates = rankingClass.getCandidateList(duplicateBugId)

        sql = 'SELECT {}.bug_id as bug_id, {}.short_desc as short_desc FROM {} WHERE bug_id={}'\
            .format(sql_project, sql_project, sql_project, duplicateBugId)
        
        db_cursor.execute(sql)

        try:
            cur_short_desc = db_cursor.fetchall()[0][1]
            predicted, predicted_summary = possible_duplicates(cur_short_desc=cur_short_desc.lower(), db_cursor=db_cursor)
                
        except IndexError:
            predicted = list()

        if i > 0 and i % 500 == 0:
            logger.info('RR calculation - {} duplicate reports were processed'.format(i))
        
        
        if len(predicted) == 0:
            logging.getLogger().warning("Bug {} has 0 candidates!".format(duplicateBugId))
            recommendation = list()
        else:
            recommendation, recommended_bugs = get_bug_ids(predicted, predicted_summary, candidates)
            
            # Optional: write to file
            # with open('./log/fts_pred_{}.log'.format(project), 'a') as f:
            #     f.write('-'*30 + '\n')
            #     f.write('Predicting for Bug - {}: {}\n'.format(duplicateBugId, cur_short_desc.lower()))
                
            #     for id, bug in zip(recommendation, recommended_bugs):
            #         f.write(id + ': ' + bug)
            #         f.write('\n')
            #     f.write('\n')

        # Update the metrics
        pos, correct_cand = recallRateMetric.update(duplicateBugId, recommendation)
        # if not correct_cand:
        #     print('dup id: ', duplicateBugId)
        #     print(cur_short_desc)
        #     print(len(predicted))
        #     print(predicted_summary[:10])
            
        positions.append(pos)
        # count += 1
        # if count == 10:
        #     sys.exit(0)

    recallRateResult = recallRateMetric.compute()

    end_time = time.time()
    nDupBugs = len(duplicateBugs)
    duration = end_time - start_time

    logger.info(
        '[Recall Rate] Throughput: {} bugs per second (bugs={} ,seconds={})'\
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


def insert_from_json_to_mysql():
    """
    insert the cleaned bug reports into mysql database
    """

    db_user = config.get('user')
    db_pwd = config.get('password')
    db_host = config.get('host')
    db_name = config.get('database')
    
    # mydb = pymysql.connect(
    #     host=db_host,
    #     user=db_user,
    #     password=db_pwd,
    #     database=db_name
    # )
    try:
        mydb = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_pwd,
            database=db_name
        )
    except mysql.connector.Error as err:
        mydb = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_pwd
        )
        my_cursor = mydb.cursor()
        my_cursor.execute("CREATE DATABASE ITS")
        mydb = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_pwd,
            database=db_name
        )
    my_cursor = mydb.cursor()

    try:
        my_cursor.execute("SELECT * FROM {}".format(sql_project))
        logging.info('table {} does not exist'.format(sql_project))
    except mysql.connector.Error as err:
        my_cursor.execute("CREATE TABLE {} (bug_id varchar(25), short_desc VARCHAR(1000))".format(sql_project))
        logging.info('created table {}'.format(sql_project))

    sql = "INSERT INTO {} (bug_id, short_desc) VALUES (%s, %s)".format(sql_project)
    with open(json_file) as f:
        lines = f.readlines()

        for line in tqdm(lines):
            cur_bug = ujson.loads(line)
            val = (cur_bug['bug_id'], cur_bug['short_desc'])
            my_cursor.execute(sql, val)
    mydb.commit()



def sql_fulltext_search(text):

    # This is as close as we can get to doing full text search using
    # standard ANSI SQL, without real full text search support. DB specific
    # modules should override this, as this will be always much slower.

    # make the string lowercase to do case insensitive search
    # lower_text = text.lower()

    # split the text we're searching for into separate words. As a hack
    # to allow quicksearch to work, if the field starts and ends with
    # a double-quote, then we don't split it into words. We can't use
    # Text::ParseWords here because it gets very confused by unbalanced
    # quotes, which breaks searches like "don't try this" (because of the
    # unbalanced single-quote in "don't").

    if len(re.findall(r"'", text)) > 0:
        text = text.replace("'", "\\'")
        words = [text]
    # if text.startswith('"') and text.endswith('"'):
    #     text = text[1:-1]
    #     words = [text]
    # words = lower_text.split()
    else:
        words = text.split()
    
    # print(words)
    # surround the words with wildcards and SQL quotes so we can use them
    # in LIKE search clauses
    
    for i in range(len(words)):
        words[i] = "'%{}%'".format(words[i])

    # print(words)
    # untaint words, since they are safe to use now that we've quoted them
    # trick_taint($_) foreach @words;

    # print(words)
    # turn the words into a set of LIKE search clauses
    for i in range(len(words)):
        words[i] = "short_desc LIKE {}".format(words[i])

    # search for occurrences of all specified words in the column
    return (" AND ".join(words), # where sql
        "CASE WHEN ("  +  " AND ".join(words) + ") THEN 1 ELSE 0 END") # relevance sql


def possible_duplicates(cur_short_desc, db_cursor):

    my_words = cur_short_desc.lower().split()

    # Remove leading/trailing punctuation from words
    for i in range(len(my_words)):
        my_words[i] = my_words[i].strip(punctuation)

    # And make sure that each word is longer than 2 characters.
    new_words = list()
    for word in my_words:
        if len(word) > 2:
            new_words.append(word)
            
    if len(new_words) == 0:
        return []
    
    if FULLTEXT_OR:
        my_joined_terms = FULLTEXT_OR.join(new_words);
        # print(my_joined_terms)
        where_sql, relevance_sql = sql_fulltext_search(my_joined_terms)
        relevance_sql = relevance_sql or where_sql
    else:
        where, relevance = list(), list()
        
        for word in new_words:
            term, rel_term = sql_fulltext_search(word)
            
            where.append(term)
            relevance.append(rel_term or term)

        where_sql = ' OR '.join(where)
        relevance_sql = ' + '.join(relevance)

    # my_possible_dupes = $dbh->selectall_arrayref(
    #     "SELECT bugs.bug_id AS bug_id, bugs.resolution AS resolution,
    #                 {relevance_sql} AS relevance
    #         FROM bugs
    #                 INNER JOIN bugs_fulltext ON bugs.bug_id = bugs_fulltext.bug_id
    #         WHERE {where_sql} {product_sql}
    #     ORDER BY {relevance} DESC, {bug_id} DESC " .sql_limit($sql_limit),
    #     {Slice => {}}
    # );
    # %s AS relevance relevance DESC

    sql = "SELECT {}.bug_id AS bug_id, {}.short_desc AS short_desc, {} AS relevance FROM {} WHERE {} ORDER BY relevance DESC, bug_id DESC".format(sql_project, sql_project, relevance_sql, sql_project, where_sql)

    # print(sql)
    db_cursor.execute(sql)
    myresult = db_cursor.fetchall()

    predicted = list()
    predicted_summary = list()
    for item in myresult:
        predicted.append(item[0])
        predicted_summary.append(item[1])
        
    # my_possible_dupes = db_cursor.execute(
    #     "SELECT {project}.bug_id AS bug_id, {relevance_sql} AS relevance
    #         FROM {project}
    #         WHERE {where_sql}
    #     ORDER BY {relevance} DESC, {bug_id} DESC "
    # );

    return predicted, predicted_summary

def make_predictions():
    db_user = config.get('user')
    db_pwd = config.get('password')
    db_host = config.get('host')
    db_name = config.get('database')

    mydb = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_pwd,
        database=db_name,
        auth_plugin='mysql_native_password'
    )
    
    with open(test_txt) as f:
        lines = f.readlines()

    dup_test_ids = lines[2].split()

    bug_report_database = BugReportDatabase.fromJson(json_file)
    my_cursor = mydb.cursor()
    recallRateDataset = BugDataset(test_txt)
    duplicateBugs = recallRateDataset.duplicateIds
    
    logRankingResult(duplicateBugs, recallRateDataset, bug_report_database, my_cursor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--project', help='project name', required=True)

    args = parser.parse_args()
    project = args.project
    
    sql_project = project
    if 'old' in project:
        sql_project = project.replace('-old', 'Old')

    log_name = None

    for i in range(1, 20):
        if not os.path.exists('./log/FTS_{}_{}.log'.format(project, i)):
            log_name = './log/FTS_{}_{}.log'.format(project, i)
            break

    logger = get_logger(log_name)
    
    test_txt = '../SABD/dataset/{}/test_{}.txt'.format(project, project)
    json_file = '../SABD/dataset/{}/{}.json'.format(project, project)
    
    insert_from_json_to_mysql()
    make_predictions()