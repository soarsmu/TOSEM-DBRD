import re
import pandas as pd
import copy
import ujson
import codecs
from tqdm import tqdm
import logging
from datetime import datetime
import logging
from collections import OrderedDict
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import pickle
from itertools import combinations 
import random
random.seed(42)

from gensim.utils import simple_preprocess


WORD_EMBEDDING_DIM = '100'
EMBEDDING_ALGO = 'sg'
HIN_EMBEDDING_DIM = '128'


def extract_bug_corpus_bugzilla(xmlfile, description):
    """
	return a list of documents from a xml file
    1) extrac bug corpora of a given bug xmfile 
        - bug summary                                
        - bug description and comments               
    2) do simple preprocess                         
        - convert a document into a list of lowercase tokens                          
        - ignore too long or too short tokens
	
	description: 1 for using, 0 not using description
    """

    f = open(xmlfile, 'r')
    contents = f.read()
    f.close()
    
    document_only_summary = []
    #find bug summary <short_desc>(.*)</short_desc>
    short_desc = re.findall('<short_desc>(.*?)</short_desc>', contents)
    if len(short_desc) != 0:
        short_desc = simple_preprocess(short_desc[0])
        bug_summary = short_desc
        document_only_summary.append(bug_summary)

    document_both_summary_description = document_only_summary
    #find bug description and comments <thetext>(.*?)</thetext>
    long_desc = re.findall('<thetext>(.*?)</thetext>', contents, re.DOTALL)
    if len(long_desc) != 0:
        for text in long_desc:
            text = simple_preprocess(text)
            document_both_summary_description.append(text)
    if description == 0:
        return document_only_summary
    else:
        return document_both_summary_description

def extract_bug_corpus_jira(xmlfile, description):
    """
	return a list of documents from a xml file
    1) extrac bug corpora of a given bug xmfile 
        - bug summary                                
        - bug description and comments               
    2) do simple preprocess                         
        - convert a document into a list of lowercase tokens                          
        - ignore too long or too short tokens
	
	description: 1 for using, 0 not using description
    """

    f = open(xmlfile, 'r')
    contents = f.read()
    f.close()
    
    document_only_summary = []
    #find bug summary <short_desc>(.*)</short_desc>
    short_desc = re.findall('<summary>(.*?)</summary>', contents)
    if len(short_desc) != 0:
        short_desc = simple_preprocess(short_desc[0])
        bug_summary = short_desc
        document_only_summary.append(bug_summary)

    document_both_summary_description = document_only_summary
    #find bug description and comments <thetext>(.*?)</thetext>
    long_desc = re.findall('<comment>(.*?)</comment>', contents, re.DOTALL)
    if len(long_desc) != 0:
        for text in long_desc:
            text = simple_preprocess(text)
            document_both_summary_description.append(text)
    if description == 0:
        return document_only_summary
    else:
        return document_both_summary_description

def extract_bug_corpus_github(jsonfile, description):
    """
	return a list of documents from a json file
    1) extrac bug corpora of a given bug jsonfile 
        - bug summary                                
        - bug description and comments               
    2) do simple preprocess                         
        - convert a document into a list of lowercase tokens                          
        - ignore too long or too short tokens
	
	description: 1 for using, 0 not using description
    """

    f = open(jsonfile, 'r')
    contents = f.read()
    f.close()
    
    document_only_summary = []
    #find bug summary <short_desc>(.*)</short_desc>
    short_desc = re.findall('<summary>(.*?)</summary>', contents)
    if len(short_desc) != 0:
        short_desc = simple_preprocess(short_desc[0])
        bug_summary = short_desc
        document_only_summary.append(bug_summary)

    document_both_summary_description = document_only_summary
    #find bug description and comments <thetext>(.*?)</thetext>
    long_desc = re.findall('<comment>(.*?)</comment>', contents, re.DOTALL)
    if len(long_desc) != 0:
        for text in long_desc:
            text = simple_preprocess(text)
            document_both_summary_description.append(text)
    if description == 0:
        return document_only_summary
    else:
        return document_both_summary_description


def random_pair_generator(number_list):
    """
    return an iterator of random pairs from a list of numbers
	"""
    used_pairs = set()
    while True:
        pair = random.sample(number_list, 2)
        pair = tuple(sorted(pair))
        if pair not in used_pairs:
            used_pairs.add(pair)
            yield pair


def generate_bug_pairs(bug_bucket_fname):
    """
    return a tuple of duplicate pair list and non-duplicate pair list
    """  

    # load bug bucket
    with open(bug_bucket_fname, 'rb') as f: 
        bug_bucket = pickle.load(f) 

    duplicate_pairs = list()

    non_duplicate_bugs = list()

    for master in bug_bucket:
        count = len(bug_bucket[master])
        if count != 0:
            duplicates = bug_bucket[master]
            duplicates.add(master)
            for pair in combinations(duplicates, 2):
                duplicate_pairs.append(pair)
        else:
            non_duplicate_bugs.append(master)

    number_duplicate_pairs = len(duplicate_pairs)

    # Number of non duplicate pair candidates
    number_non_duplicate_pairs = 4 * number_duplicate_pairs

    non_duplicate_pairs = [next(random_pair_generator(non_duplicate_bugs)) for i in range(number_non_duplicate_pairs)]

    return duplicate_pairs, non_duplicate_pairs

def get_max_sequence(project):
    project_json = '../SABD/dataset/{}/{}_soft_clean.json'.format(project, project)

    with open(project_json) as f:
        lines = f.readlines()

    num_words = []
    for line in tqdm(lines):
        cur_dict = ujson.loads(line)
        num_words.append(len(cur_dict['short_desc'].split()))
    return max(num_words)


## Pre Process and convert texts to a list of words '''
def text_to_word_list(text):
    ''' Pre Process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

# added
def save_text_hin_embeddings(PROJECT):
    WORD_EMBEDDING_FILE = 'data/pretrained_embeddings/word2vec/{}-vectors-gensim-'.format(PROJECT) + EMBEDDING_ALGO + WORD_EMBEDDING_DIM +'dwin10.bin'
    
    HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + PROJECT + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'
    
    HIN_NODE_DICT = 'data/hin_node_dict/' + PROJECT + '_node.dict'
    with open(HIN_NODE_DICT, 'r') as f:
        hin_node_dict = ujson.load(f)
        
    corpus_pkl = './data/model_training/{}_corpus.pkl'.format(PROJECT)
    data_df = pd.read_pickle(corpus_pkl)
    data_df['hin'] = 'nan'
    
    stops = set(stopwords.words('english'))
    
    ##### Prepare word embedding -- Summary
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    word2vec = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE, binary=True)

    # Iterate over the summaries
    # vocabulary: 就是一个单词对应着它的id
    # e.g., {..., '18milestones': 12845, 'netty': 12846}
    # inverse_vocabulary: 就是一个word的list，长度和vocabulary一样
    # e.g., ['<unk>', 'eclips', 'run', 'working', 'pleas']
    # for index, row in tqdm(data_df.iterrows()):
    #     # Iterate through the text of both summaries of the row
    #     s2n = []
    #     for word in text_to_word_list(row['summary']):
    #         # Check for unwanted words
    #         if word in stops and word not in word2vec.key_to_index.keys():
    #             continue

    #         if word not in vocabulary:
    #             vocabulary[word] = len(inverse_vocabulary)
    #             s2n.append(len(inverse_vocabulary))
    #             inverse_vocabulary.append(word)
    #         else:
    #             s2n.append(vocabulary[word])

    #     # Replace summaries as word to summary as number representaion
    #     data_df.at[index, 'summary'] = s2n
    
    # # print(data_df.summary)
    # # Word embedding matrix settings
    # word_embedding_dim = int(WORD_EMBEDDING_DIM)
    # word_embeddings = 1 * np.random.randn(len(vocabulary) + 1, word_embedding_dim)
    # word_embeddings[0] = 0

    # # Build the word embedding matrix
    # for word, index in vocabulary.items():
    #     if word in word2vec.key_to_index.keys():
    #         word_embeddings[index] = word2vec.word_vec(word)
            
    # # print(word_embeddings.shape)
    # del word2vec
    
    # Prepare hin embedding
    hin_vocabulary = set()
    hin_cols = ['bid', 'pro', 'com', 'ver', 'sev', 'pri']
    
    for index, row in tqdm(data_df.iterrows()):
        s2n = []
        for hin in hin_cols:
            if str(row[hin]) != 'nan' and len(str(row[hin])) > 0:
                hin_node_id = hin_node_dict[str(row[hin])][0]
                hin_vocabulary.add(hin_node_id)
                s2n.append(hin_node_id) 
            else:
                s2n.append(0)
        data_df.at[index,'hin'] = s2n
        
    # HIN embedding matrix settings
    # hin_embedding_dim = int(HIN_EMBEDDING_DIM)
    # hin_embeddings = 1 * np.random.randn(max(hin_vocabulary) + 1, hin_embedding_dim)
    # hin_embeddings[0] = 0

    # # Load hin node2vec
    # node2vec = {}
    # with open(HIN_EMBEDDING_FILE) as f:
    #     first = True
    #     for line in f:
    #         if first:
    #             first = False
    #             continue
    #         line = line.strip()
    #         tokens = line.split(' ')
    #         node2vec[tokens[0]] = np.array(tokens[1:], dtype=float)
            
    # # print(node2vec)
    # # Build the hin embedding matrix
    # for hin_node_id in hin_vocabulary:
    #     hin_embeddings[hin_node_id] = node2vec[str(hin_node_id)]


def extract_bug_pair_info(bug_pair_item, cur_project):
    items = bug_pair_item.split(',')

    # Bug 1
    bid1 = (int) (items[0])
    with codecs.open('./data/info_json/{}/{}.json'.format(cur_project, bid1), 'r', encoding='utf-8') as json1_file:
        information1 = ujson.loads(json1_file.read())

    summary1 = information1['summary']
    description1 = information1['description']
    pro1 = information1['pro']
    com1 = information1['com']
    ver1 = information1['ver']
    sev1 = information1['sev']
    pri1 = information1['pri']
    sts1 = information1['sts']


    # Bug 2
    bid2 = (int) (items[1])
    with codecs.open('./data/info_json/{}/{}.json'.format(cur_project, bid2), 'r', encoding='utf-8') as json2_file:
        information2 = ujson.loads(json2_file.read())

    summary2 = information2['summary']
    description2 = information2['description']
    pro2 = information2['pro']
    com2 = information2['com']
    ver2 = information2['ver']
    sev2 = information2['sev']
    pri2 = information2['pri']
    sts2 =  information2['sts']

    # Label
    # 1 -> dup, 0 -> no_dup
    if (int) (items[2]) == 1:
        is_duplicate = 1
    else:
        is_duplicate = 0

    return [bid1, summary1, description1, pro1, com1, ver1, sev1, pri1, sts1, \
        bid2, summary2, description2, pro2, com2, ver2, sev2, pri2, sts2, is_duplicate]
        

# hin node dictionary - {xml_object: (node_id,node_class)}
node_dict = {}

def nodeGenerationFromJson(cur_dict):
    # 对于每一个bug的dict
    #Bug ID
    BID = cur_dict['bug_id']

    #Product
    PRO = cur_dict['product']
    if len(PRO) != 0:
        PRO = 'PRO_' + PRO
    else:
        PRO = ''

    #Component
    COM = cur_dict['component']
    if len(COM) != 0:
        COM = 'COM_' + COM
    else:
        COM = ''

    #Version
    VER = cur_dict['version']
    if len(VER) != 0:
        VER = 'VER_' + VER
    else:
        VER = ''

    #Severity
    SEV = cur_dict['bug_severity']
    if len(SEV) != 0:
        SEV = 'SEV_' + SEV
    else:
        SEV = ''

    #Priority
    PRI = cur_dict['priority']
    if len(PRI) != 0:
        PRI = 'PRI_' + PRI
    else:
        PRI = ''

    nodes = [('BID', BID), ('PRO', PRO), ('COM', COM), \
        ('VER', VER), ('SEV', SEV), ('PRI', PRI)]

    for node in nodes:
        # 只把非空的放进node_dict
        if node[1] != '':
            if node[1] not in node_dict:
                # node_dict: (node_id, node_class)
                node_dict[node[1]] = (len(node_dict) + 1, node[0])
    return nodes, node_dict

#node extraction from bug report (.xml)
def nodeGeneration(xmlfile):
    f = open(xmlfile,'r')
    content = f.read()
    f.close()

    #Bug ID
    BID = re.findall('<bug_id>(.*)</bug_id>',content)[0]

    #Product
    PRO = re.findall('<product>(.*)</product>',content)
    if len(PRO) != 0:
        PRO = 'PRO_' + PRO[0]
    else:
        PRO = ''

    #Component
    COM = re.findall('<component>(.*)</component>',content)
    if len(COM) != 0:
        COM = 'COM_' + COM[0]
    else:
        COM = ''

    #Version
    if 'linux' in xmlfile:
        VER = re.findall('<cf_kernel_version>(.*)</cf_kernel_version>',content)
    else:
        
        VER = re.findall('<version>(.*)</version>',content)
    if len(VER) != 0:
        VER = 'VER_' + VER[0]
    else:
        VER = ''

    #Severity
    SEV = re.findall('<bug_severity>(.*)</bug_severity>',content)
    if len(SEV) != 0:
        SEV = 'SEV_' + SEV[0]
    else:
        SEV = ''

    #Priority
    PRI = re.findall('<priority>(.*)</priority>',content)
    if len(PRI) != 0:
        PRI = 'PRI_' + PRI[0]
    else:
        PRI = ''

    nodes = [('BID',BID),('PRO',PRO),('COM',COM),('VER',VER),('SEV',SEV),('PRI',PRI)]

    for node in nodes:
        if node[1] != '':
            if node[1] not in node_dict:
                #node_dict: (node_id, node_class)
                node_dict[node[1]] = (len(node_dict) + 1, node[0])
    
    return nodes, node_dict


#edge generation from nodes
def edgeGeneration(nodes, node_dict, option):

    #output hin format for hin2vec tool 
    if option == 'default':
        edges = []
        BID = nodes[0]
        PRO = nodes[1]
        COM = nodes[2]
        VER = nodes[3]
        SEV = nodes[4]
        PRI = nodes[5]
#        PLT = nodes[6]
#        OS  = nodes[7]

        BID_id = node_dict[BID[1]][0]
        BID_type = node_dict[BID[1]][1]

        #1 BID-COM
        if COM[1] != '':
            COM_id = node_dict[COM[1]][0]
            COM_type = node_dict[COM[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(COM_id) + '\t' + COM_type + '\t' + BID_type + '-' + COM_type + '\n'
            edges.append(edge)
            edge = str(COM_id) + '\t' + COM_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + COM_type + '-' + BID_type + '\n'
            edges.append(edge)

        #2 BID-SEV
        if SEV[1] != '':
            SEV_id = node_dict[SEV[1]][0]
            SEV_type = node_dict[SEV[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(SEV_id) + '\t' + SEV_type + '\t' + BID_type + '-' + SEV_type + '\n'
            edges.append(edge)
            edge = str(SEV_id) + '\t' + SEV_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + SEV_type + '-' + BID_type + '\n'
            edges.append(edge)

        #3 BID-PRI
        if PRI[1] != '':
            PRI_id = node_dict[PRI[1]][0]
            PRI_type = node_dict[PRI[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(PRI_id) + '\t' + PRI_type + '\t' + BID_type + '-' + PRI_type + '\n'
            edges.append(edge)
            edge = str(PRI_id) + '\t' + PRI_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + PRI_type + '-' + BID_type + '\n'
            edges.append(edge)

        #4 BID-VER
        if VER[1] != '':
            VER_id = node_dict[VER[1]][0]
            VER_type = node_dict[VER[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(VER_id) + '\t' + VER_type + '\t' + BID_type + '-' + VER_type + '\n'
            edges.append(edge)
            edge = str(VER_id) + '\t' + VER_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + VER_type + '-' + BID_type + '\n'
            edges.append(edge)

        #5 COM-PRO
        if COM[1] != '' and PRO[1] != '':
            COM_id = node_dict[COM[1]][0]
            COM_type = node_dict[COM[1]][1]
            PRO_id = node_dict[PRO[1]][0]
            PRO_type = node_dict[PRO[1]][1]
            edge = str(COM_id) + '\t' + COM_type + '\t' + str(PRO_id) + '\t' + PRO_type + '\t' + COM_type + '-' + PRO_type + '\n'
            edges.append(edge)
            edge = str(PRO_id) + '\t' + PRO_type + '\t' + str(COM_id) + '\t' + COM_type + '\t' + PRO_type + '-' + COM_type + '\n'
            edges.append(edge)

    return edges


"""
Copied and modified from MSR20 code

This class represents a bug report database where we can find all bug reports that are available.

Each dataset has bug report ids and the ids of duplicate bug reports.
"""

class BugDataset(object):

    def __init__(self, file):
        f = open(file, 'r')
        self.info = f.readline().strip()
        self.bugIds = [id for id in f.readline().strip().split()]
        self.duplicateIds = [id for id in f.readline().strip().split()]


class BugReportDatabase(object):
    '''

    Load bug report data (categorical information, summary and description) from json file.
    '''

    def __init__(self, iterator):
        self.bugById = OrderedDict()
        self.bugList = []
        self.logger = logging.getLogger()

        nEmptyDescription = 0

        for bug in iterator:
            if bug is None:
                continue

            bugId = bug["bug_id"]

            self.bugById[bugId] = bug
            self.bugList.append(bug)

            description = bug["description"]

            if isinstance(description, list) or len(description.strip()) == 0:
                nEmptyDescription += 1

        self.logger.info("Number of bugs with empty description: %d" % nEmptyDescription)

    @staticmethod
    def fromJson(fileToLoad):
        f = codecs.open(fileToLoad, 'r', encoding='utf-8')
        iterator = map(lambda line: ujson.loads(line) if len(line.strip()) > 0 else None, f)
        return BugReportDatabase(iterator)

    def getBug(self, bugId):
        return self.bugById[bugId]

    def getBugByIndex(self, idx):
        return self.bugList[idx]

    def __len__(self):
        return len(self.bugList)
    
    def __contains__(self, bug):
        bugId = bug['bug_id'] if isinstance(bug, dict) else bug

        return bugId in self.bugById

    def getMasterIdByBugId(self, bugs=None):
        # return the dup_id by bug_id
        masterIdByBugId = {}
        bugs = self.bugList if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.bugById[bug]

            bugId = bug['bug_id']
            dupId = bug['dup_id']

            if len(dupId) != 0:
                masterIdByBugId[bugId] = dupId
            else:
                masterIdByBugId[bugId] = bugId

        return masterIdByBugId

    def getMasterSetById(self, bugs=None):
        masterSetById = {}
        bugs = self.bugList if bugs is None else bugs

        for bug in bugs:
            if not isinstance(bug, dict):
                bug = self.bugById[bug]

            dupId = bug['dup_id']

            if len(dupId) != 0:
                masterSet = masterSetById.get(dupId, set())

                if len(masterSet) == 0:
                    masterSetById[dupId] = masterSet

                masterSet.add(bug['bug_id'])

        # Insert id of the master bugs in your master sets
        for masterId, masterSet in masterSetById.items():
            if masterId in self:
                masterSet.add(masterId)

        return masterSetById


def readDateFromBug(bug):
    return datetime.strptime(bug['creation_ts'], '%Y-%m-%d %H:%M:%S %z')

class SunRanking(object):
    def __init__(self, bugReportDatabase, dataset, window):
        self.bugReportDatabase = bugReportDatabase
        self.masterIdByBugId = self.bugReportDatabase.getMasterIdByBugId()
        self.duplicateBugs = dataset.duplicateIds
        self.candidates = []
        self.window = int(window) if window is not None else 0
        self.latestDateByMasterSetId = {}
        self.logger = logging.getLogger()

        # Get oldest and newest duplicate bug report in dataset
        oldestDuplicateBug = (
            self.duplicateBugs[0], 
            readDateFromBug(self.bugReportDatabase.getBug(self.duplicateBugs[0]))
        )

        for dupId in self.duplicateBugs:
            dup = self.bugReportDatabase.getBug(dupId)
            creationDate = readDateFromBug(dup)

            if oldestDuplicateBug[1] < creationDate:
                oldestDuplicateBug = (dupId, creationDate)

        # Keep only master that are able to be candidate
        for bug in self.bugReportDatabase.bugList:
            bugCreationDate = readDateFromBug(bug)
            bugId = bug['bug_id']

            # Remove bugs that their creation time is bigger than oldest duplicate bug
            if bugCreationDate > oldestDuplicateBug[1] or (
                    bugCreationDate == oldestDuplicateBug[1] and \
                        bug['bug_id'] > oldestDuplicateBug[0]):
                continue

            self.candidates.append((bugId, bugCreationDate.timestamp()))

        # Keep the timestamp of all reports in each master set
        for masterId, masterSet in self.bugReportDatabase.getMasterSetById(
                map(lambda c: c[0], self.candidates)).items():
            ts_list = []

            for bugId in masterSet:
                bugCreationDate = readDateFromBug(self.bugReportDatabase.getBug(bugId))

                ts_list.append((int(bugId), bugCreationDate.timestamp()))

            self.latestDateByMasterSetId[masterId] = ts_list

        # Set all bugs that are going to be used by our models.
        self.allBugs = [bugId for bugId, bugCreationDate in self.candidates]
        self.allBugs.extend(self.duplicateBugs)

    def getDuplicateBugs(self):
        return self.duplicateBugs

    def getAllBugs(self):
        return self.allBugs

    def getCandidateList(self, anchorId):
        candidates = []
        anchor = self.bugReportDatabase.getBug(anchorId)
        anchorCreationDate = readDateFromBug(anchor)
        anchorMasterId = self.masterIdByBugId[anchorId]
        nDupBugs = 0
        anchorTimestamp = anchorCreationDate.timestamp()
        anchorDayTimestamp = int(anchorTimestamp / (24 * 60 * 60))

        nSkipped = 0
        window_record = [] if self.logger.isEnabledFor(logging.DEBUG) else None
        anchorIdInt = int(anchorId)

        for bugId, bugCreationDate in self.candidates:
            bugIdInt = int(bugId)

            # Ignore reports that were created after the anchor report
            if bugCreationDate > anchorTimestamp or (
                    bugCreationDate == anchorTimestamp and bugIdInt > anchorIdInt):
                continue

            # Check if the same report
            if bugId == anchorId:
                continue

            if bugIdInt > anchorIdInt:
                self.logger.warning(
                    "Candidate - consider a report which its id {} is bigger than duplicate {}".format(bugId, anchorId)
                )

            masterId = self.masterIdByBugId[bugId]

            # Group all the duplicate and master in one unique set. 
            # Creation date of newest report is used to filter the bugs
            tsMasterSet = self.latestDateByMasterSetId.get(masterId)

            if tsMasterSet:
                max = -1
                newest_report = None

                for candNewestId, ts in self.latestDateByMasterSetId[masterId]:
                    # Ignore reports that were created after the anchor or 
                    # the ones that have the same ts and bigger id
                    if ts > anchorTimestamp or (ts == anchorTimestamp and candNewestId >= anchorIdInt):
                        continue

                    if candNewestId >= anchorIdInt:
                        self.logger.warning(
                            "Window filtering - consider a report which its id {} is bigger than duplicate {}".format(candNewestId, anchorIdInt)
                        )

                    # Get newest ones
                    if max < ts:
                        max = ts
                        newest_report = candNewestId

                # Transform to day timestamp
                bug_timestamp = int(max / (24 * 60 * 60))
            else:
                # Transform to day timestamp
                bug_timestamp = int(bugCreationDate / (24 * 60 * 60))
                newest_report = bugId

            # Is it in the window?
            if 0 < self.window < (anchorDayTimestamp - bug_timestamp):
                nSkipped += 1
                continue

            # Count number of duplicate bug reports
            if anchorMasterId == masterId:
                nDupBugs += 1

            # It is a candidate
            candidates.append(bugId)
            if window_record is not None:
                window_record.append((bugId, newest_report, bug_timestamp))

        self.logger.debug(
            "Query {} ({}) - window {} - number of reports skipped: {}" \
                .format(anchorId, anchorDayTimestamp, self.window, nSkipped)
        )

        if window_record is not None:
            self.logger.debug("{}".format(window_record))

        if nDupBugs == 0:
            return []

        return candidates

def fit_tokenizer(data_pkl):
    """
    fit a tokenzier on the data
    """

    from tensorflow.keras.preprocessing.text import Tokenizer

    data_df = pd.read_pickle(data_pkl)
    tokenzier = Tokenizer(oov_token="<unk>")
    text = data_df['summary'].to_numpy()
    text = np.asarray(text).astype(str)
    tokenzier.fit_on_texts(text)
    return tokenzier

    # from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    # import tensorflow as tf
    # data_df = pd.read_pickle(data_pkl)
    # text = data_df['summary'].to_numpy()
    # text = np.asarray(text).astype(str)
    # vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=100)
    # text_ds = tf.data.Dataset.from_tensor_slices(text).batch(128)
    # vectorizer.adapt(text_ds)
    # return vectorizer

def prepare_word_embedding_matrix(tokenizer, cur_project):
    """
    use the vectorizer to get a word embedding matrix
    整个模型也只需要一个
    """

    WORD_EMBEDDING_FILE = 'data/pretrained_embeddings/word2vec/{}-vectors-gensim-'.format(cur_project) + EMBEDDING_ALGO + WORD_EMBEDDING_DIM +'dwin10.bin'
    
    word2vec = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE, binary=True)
    logging.info("Found %s word vectors." % len(word2vec))

    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    # voc = tokenizer.get_vocabulary()
    # word_index = dict(zip(voc, range(len(voc))))

    voc = tokenizer.word_index
    word_index = dict(tokenizer.word_index)
    num_tokens = len(voc) + 2

    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    # for word, i in tokenizer.word_index.items():
    for word, i in word_index.items():
        try:
            embedding_vector = word2vec[word]
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        except KeyError:
            misses += 1
    logging.info("Converted %d words (%d misses)" % (hits, misses))
    
    return embedding_matrix


def prepare_hin_embedding_matrix(data_pkl, cur_project):
    # 整个model只需要一个embedding matrix
    # hin_vocabulary是用来设定hin_embedding_matrix的大小的
    
    HIN_NODE_DICT = 'data/hin_node_dict/' + cur_project + '_node.dict'
    HIN_EMBEDDING_FILE = HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + cur_project + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'

    data_df = pd.read_pickle(data_pkl)

    # Initialize hin features
    data_df['hin'] = 'nan'

    logging.info('built word embeddings\n starting prepare hin embedding')

    # Prepare hin embedding
    hin_vocabulary = set()

    hin_cols = ["bid", "pro", "com", "ver", "sev", "pri"]

    with open(HIN_NODE_DICT, 'r') as f:
        hin_node_dict = ujson.load(f)

    for index, row in data_df.iterrows():
        s2n = []
        for hin in hin_cols:
            if str(row[hin]) != 'nan' and len(str(row[hin])) > 0:
                hin_node_id = hin_node_dict[str(row[hin])][0]
                hin_vocabulary.add(hin_node_id)
                s2n.append(hin_node_id)
            else:
                s2n.append(0)
        data_df.at[index, 'hin'] = s2n

    # HIN embedding matrix settings
    hin_embedding_dim = int(HIN_EMBEDDING_DIM)

    hin_embeddings = 1 * np.random.randn(max(hin_vocabulary) + 1, hin_embedding_dim)
    hin_embeddings[0] = 0

    # Load hin node2vec
    node2vec = {}
    with open(HIN_EMBEDDING_FILE) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            line = line.strip()
            tokens = line.split(' ')
            node2vec[tokens[0]] = np.array(tokens[1:],dtype=float)

    for hin_node_id in hin_vocabulary:
        hin_embeddings[hin_node_id] = node2vec[str(hin_node_id)]

    return hin_embeddings


def transform_hin_ids(data_csv, cur_project):
    """
    transform to hin representations
    """

    HIN_NODE_DICT = 'data/hin_node_dict/' + cur_project + '_node.dict'
    with open(HIN_NODE_DICT, 'r') as f:
        hin_node_dict = ujson.load(f)
        
    data_df = pd.read_csv(data_csv)

    # Initialize hin features
    data_df['hin1'] = 'nan'
    data_df['hin2'] = 'nan'

    hin_cols1 = ["bid1", "pro1", "com1", "ver1", "sev1", "pri1"]
    hin_cols2 = ["bid2", "pro2", "com2", "ver2", "sev2", "pri2"]

    for index, row in data_df.iterrows():
        s2n = []
        for hin in hin_cols1:
            if str(row[hin]) != 'nan' and len(str(row[hin])) > 0:
                hin_node_id = hin_node_dict[str(row[hin])][0]
                s2n.append(hin_node_id) 
            else:
                s2n.append(0)
        data_df.at[index,'hin1'] = s2n

        s2n = []
        for hin in hin_cols2:
            if str(row[hin]) != 'nan' and len(str(row[hin])) > 0:
                hin_node_id = hin_node_dict[str(row[hin])][0]
                s2n.append(hin_node_id) 
            else:
                s2n.append(0)
        data_df.at[index,'hin2'] = s2n

    return data_df['hin1'], data_df['hin2']


def save_corpus_ids(data_pkl, tokenizer, max_len, cur_project):
    logging.info('convert corpus info to ids...')
    # 每一行存的是text和hin
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    data_df = pd.read_pickle(data_pkl)
    data_df['hin'] = 'nan'
    hin_cols = ["bid", "pro", "com", "ver", "sev", "pri"]
    
    HIN_NODE_DICT = 'data/hin_node_dict/' + cur_project + '_node.dict'
    with open(HIN_NODE_DICT, 'r') as f:
        hin_node_dict = ujson.load(f)
    # convert hin ids
    for index, row in data_df.iterrows():
        s2n = []
        for hin in hin_cols:
            if str(row[hin]) != 'nan' and len(str(row[hin])) > 0:
                hin_node_id = hin_node_dict[str(row[hin])][0]
                s2n.append(hin_node_id) 
            else:
                s2n.append(0)
        data_df.at[index, 'hin'] = s2n
    
    data_df['padded'] = 'nan'
    # convert text ids
    sum_seq = tokenizer.texts_to_sequences(data_df.summary)
    padded = pad_sequences(sum_seq, maxlen=max_len)
    for index in range(padded.shape[0]):
        data_df.at[index, 'padded'] = padded[index]
        
    bid_representations = {}
    for index, row in data_df.iterrows():
        bid_representations[row['bid']] = {
            'hin': row['hin'],
            'summary': row['padded']}
    
    with open('data/model_training/{}_bid_corpus.pkl'.format(cur_project), 'wb') as f:
        pickle.dump(bid_representations, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    new_df = pd.DataFrame({
        'hin': data_df['hin'],
        'summary': data_df['padded']})
    new_df.to_pickle('data/model_training/{}_corpus_ids.pkl'.format(cur_project))
    
    print(new_df.head())
    print(new_df.shape)
    
def convert_train_valid_to_ids(text_file, cur_project, file_name):
    with open('data/model_training/{}_bid_corpus.pkl'.format(cur_project), 'rb') as f:
        bid_representations = pickle.load(f)
        
    with open(text_file) as f:
        lines = f.readlines()
    
    df = pd.DataFrame()
    df['text_left'] = 'nan'
    df['text_right'] = 'nan'
    df['hin_left'] = 'nan'
    df['hin_right'] = 'nan'
    df['is_duplicate'] = 'nan'
    
    for index in tqdm(range(len(lines))):
        bug_1, bug_2, is_dup = lines[index].strip().split(',')
        if int(is_dup) > 0:
            df.at[index, 'is_duplicate'] = 1
        else:
            df.at[index, 'is_duplicate'] = 0
        df.at[index, 'text_left'] = bid_representations[bug_1]['summary']
        df.at[index, 'text_right'] = bid_representations[bug_2]['summary']
        df.at[index, 'hin_left'] = bid_representations[bug_1]['hin']
        df.at[index, 'hin_right'] = bid_representations[bug_2]['hin']
    
    df.to_pickle(file_name)