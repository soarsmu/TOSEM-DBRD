"""
Generate the data to train and test REP and BM25F_EXT

Example:

python classical_approach/generate_input_dbrd.py --database dataset/eclipseInitial/eclipseInitial.json --test dataset/eclipseInitial/test_eclipseInitial.txt --output dataset/eclipseInitial/dbrd_test.txt

python classical_approach/generate_input_dbrd.py --database dataset/mozillaInitial/mozillaInitial.json --test dataset/mozillaInitial/test_mozillaInitial.txt --output dataset/mozillaInitial/dbrd_test.txt

"""


import argparse
import codecs
import logging
import re
import string
from collections import Counter
import os

from nltk import SnowballStemmer
from nltk.corpus import stopwords

import sys
sys.path.append('.')

from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.preprocessing import ClassicalPreprocessing, MultiLineTokenizer, \
    HTMLSymbolFilter



class DBRDPreprocessing(ClassicalPreprocessing):
    REGEX = re.compile('[\W_]+', re.UNICODE)

    def __init__(self):
        super(DBRDPreprocessing, self).__init__(MultiLineTokenizer(), SnowballStemmer('english', ignore_stopwords=True), set(stopwords.words('english') + list(string.punctuation) + ["n't", "'t"]),
        [HTMLSymbolFilter()])

    def preprocess(self, text):
        tokens = super(DBRDPreprocessing, self).preprocess(text)

        # Remove numbers and punctuantions
        preprocessedtokens = []

        for tok in tokens:
            # Split tokens by non-alhpanumeric characters
            split_punc_token = re.split(DBRDPreprocessing.REGEX, tok)
            # Remove numbers
            preprocessedtokens.extend(filter(lambda t: len(t) > 0 and re.match("[0-9]+$", t) is None, split_punc_token))

        return preprocessedtokens


def ngram_function(n, tokens):
    if len(tokens) < n:
        return

    for i in range(len(tokens) - (n - 1)):
        yield " ".join(tokens[i:i + n])


class CoutBoW(object):

    def __init__(self):
        self.vocab = {}

    def freq(self, tokens, ngram):
        if ngram == 1:
            ngram_tokens = iter(tokens)
        else:
            ngram_tokens = ngram_function(ngram, tokens)

        token_ids = []
        ngrams = []

        for tok in ngram_tokens:
            tok_id = self.vocab.get(tok)

            if tok_id is None:
                tok_id = len(self.vocab)
                self.vocab[tok] = tok_id

            token_ids.append(tok_id)
            ngrams.append(tok)

        return Counter(token_ids).most_common(), ngrams


def format_tf_to_text(tf_list):
    return ",".join(["%d:%d" % (id, tf) for id, tf in sorted(tf_list)])


def generate_input_vec(database, max_bug_id):
    bow = CoutBoW()
    logger = logging.getLogger()

    version_set = set()
    product_dict = {}
    component_dict = {}
    type_dict = {}  # enhacement 1 others 0 (bug_severity)
    priority_set = set()

    preprocessed_bugs = []
    preprocessing = DBRDPreprocessing()

    for bug_id, bug in sorted(map(lambda item: (int(item[0]), item[1]), database.bugById.items())):
        if bug_id > max_bug_id:
            continue

        sum = bug['short_desc']
        desc = bug['description']
        comp = bug['component']
        version = bug['version']
        product = bug['product']
        priority = bug['priority']
        severity = bug['bug_severity']

        preprocessed_sum = preprocessing.preprocess(sum)
        preprocessed_desc = preprocessing.preprocess(desc)

        version_set.add(version)
        product_dict.setdefault(product, len(product_dict))
        component_dict.setdefault(comp, len(component_dict))
        priority_set.add(priority)
        type_dict.setdefault(severity, len(type_dict))

        if len(preprocessed_sum) == 0:
            logger.info("A bug {} (dup_id={})  has a empty summary".format(bug_id, bug['dup_id']))
            preprocessed_sum = ["#EMPTY"]

        if len(preprocessed_desc) == 0:
            logger.info("A bug {} (dup_id={}) has a empty description".format(bug_id, bug['dup_id']))
            preprocessed_desc = ["#EMPTY"]

        bug['sum_uni'], sum_uni = bow.freq(preprocessed_sum, 1)
        bug['sum_bi'], sum_bi = bow.freq(preprocessed_sum, 2)
        bug['sum_tri'], sum_tri = bow.freq(preprocessed_sum, 3)

        bug['desc_uni'], desc_uni = bow.freq(preprocessed_desc, 1)
        bug['desc_bi'], desc_bi = bow.freq(preprocessed_desc, 2)
        bug['desc_tri'], desc_tri = bow.freq(preprocessed_desc, 3)

        bug['total_uni'], _ = bow.freq(sum_uni + desc_uni, 1)
        bug['total_bi'], _ = bow.freq(sum_bi + desc_bi, 1)
        bug['total_tri'], _ = bow.freq(sum_tri + desc_tri, 1)

        preprocessed_bugs.append(bug)

    version_dict = dict([(k, i) for i, k in enumerate(sorted(version_set))])
    priority_dict = dict([(k, i) for i, k in enumerate(sorted(priority_set))])
    logger.info(version_dict)
    logger.info(priority_dict)
    logger.info(product_dict)
    logger.info(component_dict)
    logger.info(type_dict)

    d = []

    for bug in preprocessed_bugs:
        out = {}

        out['id'] = bug['bug_id']

        out['S-U'] = sorted(bug['sum_uni'])
        out['S-B'] = sorted(bug['sum_bi'])
        out['S-T'] = sorted(bug['sum_tri'])

        out['D-U'] = sorted(bug['desc_uni'])
        out['D-B'] = sorted(bug['desc_bi'])
        out['D-T'] = sorted(bug['desc_tri'])

        out['A-U'] = sorted(bug['total_uni'])
        out['A-B'] = sorted(bug['total_bi'])
        out['A-T'] = sorted(bug['total_tri'])

        out['DID'] = '' if len(bug['dup_id']) == 0 else bug['dup_id']
        out['VERSION'] = version_dict[bug['version']]
        out['COMPONENT'] = product_dict[bug['product']]
        out['SUB-COMPONENT'] = component_dict[bug['component']]
        out['TYPE'] = type_dict[bug['bug_severity']]
        out['PRIORITY'] = priority_dict[bug['priority']]

        d.append(out)

    return d, max(bow.vocab.values())


def generate_file(reports, output_path, black_list=set()):
    out_file = codecs.open(output_path, 'w')

    for rep in reports:
        bug_id = rep['id']
        if len(black_list) > 0 and str(bug_id) in black_list:
            continue

        out_file.write("ID={}\n".format(rep['id']))

        out_file.write("S-U={}\n".format(format_tf_to_text(rep['S-U'])))
        out_file.write("S-B={}\n".format(format_tf_to_text(rep['S-B'])))
        out_file.write("S-T={}\n".format(format_tf_to_text(rep['S-T'])))

        out_file.write("D-U={}\n".format(format_tf_to_text(rep['D-U'])))
        out_file.write("D-B={}\n".format(format_tf_to_text(rep['D-B'])))
        out_file.write("D-T={}\n".format(format_tf_to_text(rep['D-T'])))

        out_file.write("A-U={}\n".format(format_tf_to_text(rep['A-U'])))
        out_file.write("A-B={}\n".format(format_tf_to_text(rep['A-B'])))
        out_file.write("A-T={}\n".format(format_tf_to_text(rep['A-T'])))

        out_file.write("DID={}\n".format('' if len(rep['DID']) == 0 else rep['DID']))
        out_file.write("VERSION={}\n".format(rep['VERSION']))
        out_file.write("COMPONENT={}\n".format(rep['COMPONENT']))
        out_file.write("SUB-COMPONENT={}\n".format(rep['SUB-COMPONENT']))
        out_file.write("TYPE={}\n".format(rep['TYPE']))
        out_file.write("PRIORITY={}\n".format(rep['PRIORITY']))


def generate_input(database, max_bug_id, output_path, black_list=set()):
    bow = CoutBoW()
    logger = logging.getLogger()

    version_set = set()
    product_dict = {}
    component_dict = {}
    type_dict = {}  # enhacement 1 others 0 (bug_severity)
    priority_set = set()

    preprocessed_bugs = []
    preprocessing = DBRDPreprocessing()

    for bug in database.bugList:
        bug_id = bug['bug_id']

        if int(bug_id) > max_bug_id:
            continue

        if len(black_list) > 0 and bug_id in black_list:
            continue

        sum = bug['short_desc']
        desc = bug['description']
        comp = bug['component']
        version = bug['version']
        product = bug['product']
        priority = bug['priority']
        severity = bug['bug_severity']

        preprocessed_sum = preprocessing.preprocess(sum)
        preprocessed_desc = preprocessing.preprocess(desc)

        version_set.add(version)
        product_dict.setdefault(product, len(product_dict))
        component_dict.setdefault(comp, len(component_dict))
        priority_set.add(priority)
        type_dict.setdefault(severity, len(type_dict))

        if len(preprocessed_sum) == 0:
            logger.info("A bug {} (dup_id={})  has a empty summary".format(bug_id, bug['dup_id']))
            preprocessed_sum = ["#EMPTY"]

        if len(preprocessed_desc) == 0:
            logger.info("A bug {} (dup_id={}) has a empty description".format(bug_id, bug['dup_id']))
            preprocessed_desc = ["#EMPTY"]

        bug['sum_uni'], sum_uni = bow.freq(preprocessed_sum, 1)
        bug['sum_bi'], sum_bi = bow.freq(preprocessed_sum, 2)
        bug['sum_tri'], sum_tri = bow.freq(preprocessed_sum, 3)

        bug['desc_uni'], desc_uni = bow.freq(preprocessed_desc, 1)
        bug['desc_bi'], desc_bi = bow.freq(preprocessed_desc, 2)
        bug['desc_tri'], desc_tri = bow.freq(preprocessed_desc, 3)

        bug['total_uni'], _ = bow.freq(sum_uni + desc_uni, 1)
        bug['total_bi'], _ = bow.freq(sum_bi + desc_bi, 1)
        bug['total_tri'], _ = bow.freq(sum_tri + desc_tri, 1)

        preprocessed_bugs.append(bug)
    version_dict = dict([(k, i) for i, k in enumerate(sorted(version_set))])
    priority_dict = dict([(k, i) for i, k in enumerate(sorted(priority_set))])
    logger.info(version_dict)
    logger.info(priority_dict)
    logger.info(product_dict)
    logger.info(component_dict)
    logger.info(type_dict)

    out_file = codecs.open(output_path, 'w')

    for bug in preprocessed_bugs:
        out_file.write("ID={}\n".format(bug['bug_id']))

        out_file.write("S-U={}\n".format(format_tf_to_text(bug['sum_uni'])))
        out_file.write("S-B={}\n".format(format_tf_to_text(bug['sum_bi'])))
        out_file.write("S-T={}\n".format(format_tf_to_text(bug['sum_tri'])))

        out_file.write("D-U={}\n".format(format_tf_to_text(bug['desc_uni'])))
        out_file.write("D-B={}\n".format(format_tf_to_text(bug['desc_bi'])))
        out_file.write("D-T={}\n".format(format_tf_to_text(bug['desc_tri'])))

        out_file.write("A-U={}\n".format(format_tf_to_text(bug['total_uni'])))
        out_file.write("A-B={}\n".format(format_tf_to_text(bug['total_bi'])))
        out_file.write("A-T={}\n".format(format_tf_to_text(bug['total_tri'])))

        out_file.write("DID={}\n".format('' if len(bug['dup_id']) == 0 else bug['dup_id']))
        out_file.write("VERSION={}\n".format(version_dict[bug['version']]))
        out_file.write("COMPONENT={}\n".format(product_dict[bug['product']]))
        out_file.write("SUB-COMPONENT={}\n".format(component_dict[bug['component']]))
        out_file.write("TYPE={}\n".format(type_dict[bug['bug_severity']]))
        out_file.write("PRIORITY={}\n".format(priority_dict[bug['priority']]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--database', required=True, help="")
    parser.add_argument('--test', required=True, help="")
    parser.add_argument('--output', required=True, help="")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    logger.addHandler(logging.StreamHandler())
    PROJECT = os.path.split(args.database)[1].split('.')[0]
    fileHandler = logging.FileHandler('./log/generate_input_dbrd_{}.log'.format(PROJECT))
    logger.addHandler(fileHandler)
    logger.info(args)

    output_path = args.output
    database = BugReportDatabase.fromJson(args.database)
    test = BugDataset(args.test)

    max_bug_id = max(map(lambda bug_id: int(bug_id), test.bugIds))

    generate_input(database, max_bug_id, output_path)
