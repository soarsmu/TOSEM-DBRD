"""
Create the dataset to train the word embeddings
python util/create_dataset_word_embedding.py --mongo_db openOffice --collection initial --training DATASET_DIR/sun_2011/open_office_2001-2008_2010/training_split_open_office.txt --bug_database DATASET_DIR/sun_2011/open_office_2001-2008_2010/open_office_initial.json --output HOME/word_embedding/open_office_2001-2008_2010_soft_clean.txt --clean_type soft --rm_punc --sent_tok --rm_number --lower_case

Use the following script to the word embedding using glove:
#!/bin/bash

CORPUS=HOME/word_embedding/open_office_2001-2008_2010_soft_clean.txt
VOCAB_FILE=vocab_open_office_80_20_soft_clean.txt
COOCCURRENCE_FILE=cooccurrence_open_office_soft_clean.bin
COOCCURRENCE_SHUF_FILE=cooccurrence_open_office_soft_clean.shuf.bin
BUILDDIR=build
SAVE_FILE=HOME/word_embedding/glove_300d_open_office_2001-2008_2010
VERBOSE=2
MEMORY=12.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=30
WINDOW_SIZE=15
BINARY=0
NUM_THREADS=8


$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
"""
import argparse
import codecs
import logging

import pymongo
from nltk import word_tokenize, sent_tokenize, WhitespaceTokenizer

from classical_approach.generate_input_dbrd import DBRDPreprocessing
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.preprocessing import loadFilters, \
    cleanDescription, softClean

from util.data_util import readDateFromBug


def applyFilters(filters, tokens, sentence):
    out = []

    for token in tokens:
        for filter in filters:
            token = filter.filter(token, sentence)

        if len(token) == 0:
            continue

        out.append(token)

    return out


parser = argparse.ArgumentParser()

# Global arguments
parser.add_argument('--mongo_db', help="Database that contains the bug reports")
parser.add_argument('--collection', help="Collection ")
parser.add_argument('--training', required=True, help="")
parser.add_argument('--bug_database', required=True, help="")
parser.add_argument('--output', required=True, help="The word embedding file with new words.")
parser.add_argument('--filters', nargs='+', help='Filter Names to be used in the summary', default=['TransformLowerCaseFilter', 'TransformNumberToZeroFilter'])
parser.add_argument('--clean_type', help="agg or soft")
parser.add_argument('--rm_punc', action="store_true")
parser.add_argument('--rm_number', action="store_true")
parser.add_argument('--sent_tok', action="store_true")
parser.add_argument('--stop_words', action="store_true")
parser.add_argument('--stem', action="store_true")
parser.add_argument('--lower_case', action="store_true")
parser.add_argument('--rm_char', action="store_true")

# Parsing
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Loading bug reports")
bugReportDatabase = BugReportDatabase.fromJson(args.bug_database)

logger.info("Loading training dataset")
training = BugDataset(args.training)

limitDate = readDateFromBug(bugReportDatabase.getBug(training.bugIds[-1]))

# Connect to mongo dataset
client = pymongo.MongoClient()
database = client[args.mongo_db]
col = database[args.collection]
filters = loadFilters(args.filters)

if args.clean_type == 'agg':
    logger.info("Aggressive clean")
    cleanFunc = cleanDescription
    space_sent_tkn = False
    tknFunct = word_tokenize
elif args.clean_type == 'soft':
    logger.info("Soft clean")
    cleanFunc = softClean
    space_sent_tkn = True
    wTkn = WhitespaceTokenizer()
    tknFunct = wTkn.tokenize
elif args.clean_type == 'rep':
    logger.info("REP clean")
    dbrd = DBRDPreprocessing()

    def clean_dbrd(text):
        return " ".join(dbrd.preprocess(text))

    cleanFunc = clean_dbrd
    space_sent_tkn = True
    wTkn = WhitespaceTokenizer()
    tknFunct = wTkn.tokenize
else:
    logger.info("No clean type")
    cleanFunc = None
    space_sent_tkn = False
    tknFunct = word_tokenize

outFile = codecs.open(args.output, 'w')
firstBugNotUsed = None

for bug in col.find({}).max_time_ms(10000):
    creationDate = readDateFromBug(bug)

    if creationDate > limitDate:
        if firstBugNotUsed is None:
            firstBugNotUsed = bug['bug_id']
            print("First bug not used: %s" % firstBugNotUsed)

        continue

    print(bug['bug_id'])

    if bug['bug_id'] == '111262':
        print("Skip")
        continue

    if int(bug['bug_id']) > int(training.bugIds[-1]):
        print("Using %s which has bigger id than %s" % (bug['bug_id'], training.bugIds[-1]))


    sumText = cleanFunc(bug['short_desc'], args.rm_punc, args.sent_tok, args.rm_number, args.stop_words, args.stem, args.lower_case, args.rm_char) if cleanFunc else bug['short_desc']
    summTokens = word_tokenize(sumText)

    if len(summTokens) > 5 and len(bug['short_desc']) > 15:
        summary = ' '.join(applyFilters(filters, summTokens, bug['short_desc']))

        outFile.write(summary)
        outFile.write('\n')

    description = bug.get('description', "")

    if len(description) > 0:
        descText = cleanFunc(bug['description'], args.rm_punc, args.sent_tok, args.rm_number, args.stop_words, args.stem, args.lower_case, args.rm_char) if cleanFunc else bug['description']

        if space_sent_tkn:
            sentences = descText.split('\n')
        else:
            sentences = sent_tokenize(descText)

        for sent in sentences:
            tokens = tknFunct(sent)
            if len(tokens) < 4:
                continue

            descLine = ' '.join(applyFilters(filters, tokens, sent))

            outFile.write(descLine)
            outFile.write('\n')

outFile.close()