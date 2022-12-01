import json
import logging
from datetime import datetime

from data.Lexicon import Lexicon
import data.preprocessing


def readDateFromBug(bug):
    return datetime.strptime(bug['creation_ts'], '%Y-%m-%d %H:%M:%S %z')

def readDateFromBugWithoutTimezone(bug):
    return datetime.strptime(bug['creation_ts'][:-6], '%Y-%m-%d %H:%M:%S')

def createChunks(l, n):
    chunkSize = int(len(l) / n)
    remaining = len(l) % n
    chunks = []
    begin = 0

    for i in range(n):
        if remaining != 0:
            additional = 1
            remaining -= 1
        else:
            additional = 0

        end = begin + chunkSize + additional
        chunks.append(l[begin:end])
        begin = end

    return chunks


class ReOrderedList(object):

    def __init__(self, arr, idxs):
        self.arr = arr
        self.idxs = idxs
        self.current = 0

    def __len__(self):
        return len(self.idxs)

    def __iter__(self):
        return self

    def __getitem__(self, item):
        return self.arr[self.idxs[item]]

    def __next__(self):
        if self.current == len(self.idxs):
            self.current = 0
            raise StopIteration

        v = self.arr[self.idxs[self.current]]
        self.current += 1

        return v


def createCategoricalPreprocessorAndLexicons(lexiconPath, bugReportDatabase):
    # Define Filters and set preprocessing steps
    basicFilter = [
        data.preprocessing.TransformLowerCaseFilter(),
    ]

    lexiconJsons = json.load(open(lexiconPath))
    productLexicon = Lexicon.fromList(lexiconJsons['product'], True, 'product')
    severityLexicon = Lexicon.fromList(lexiconJsons['bug_severity'], True, 'bug_severity')
    componentLexicon = Lexicon.fromList(lexiconJsons['component'], True, 'component')
    priorityLexicon = Lexicon.fromList(lexiconJsons['priority'], True, 'priority')

    categoricalArgs = [
        ('product', productLexicon, basicFilter),
        ('bug_severity', severityLexicon, basicFilter),
        ('component', componentLexicon, basicFilter),
        ('priority', priorityLexicon, basicFilter),
        # BasicFieldPreprocessor('version', versionLexicon, basicFilter + [TransformNumberToZeroFilter()]),
    ]

    str = "Field name and Lexicon size: "

    for f, l, _ in categoricalArgs:
        str += "{} {}; ".format(f, l.getLen())

    logging.getLogger().info(str)

    lexicons = [productLexicon, severityLexicon, componentLexicon, priorityLexicon]

    return data.preprocessing.CategoricalPreprocessor(categoricalArgs, bugReportDatabase), lexicons
