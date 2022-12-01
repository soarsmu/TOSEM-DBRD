#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters are classes that change the token, like transform to lower case the letters 
"""

import hashlib
import importlib
import itertools
import logging
import os
import pickle
import re
import string


from nltk import WhitespaceTokenizer, SnowballStemmer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import sent_tokenize


CLASS_DEF_JAVA_REGEX = r'((public|private)\s*)?class\s+(\w+)(\s+extends\s+(\w+))?(\s+implements\s+([\w,\s]+))?\s+\{(\s*})?'
FUNC_IF_DEF_JAVA_REGEX = r'\w+\s*\([^)]*\)\s*\{'
IF_JAVA_REGEX = r'if\s*\(.*\{'
OBJ_JAVA_REGEX = r'\w\s*=\snew[^;]+;'

SPLIT_PUNCT_REGEX = r"([\"!#\\$%&()*+,-./:;<=>?@[\]^_'`{|}~])"
RVM_REPEATED_PUNC = "([%s])\\1{1,}" % string.punctuation


def checkDesc(desc):
    return desc.strip() if desc and len(desc) > 0 else ""

# Filters
class Filter:

    def filter(self, token, sentence):
        raise NotImplementedError()

    def getSymbol(self):
        return None


class TransformLowerCaseFilter(Filter):
    def filter(self, token, sentence):
        return token.lower()


class TransformNumberToZeroFilter(Filter):
    def filter(self, token, sentence):
        return re.sub('[0-9]', '0', token)


class MultiLineTokenizer(object):
    """Tokenize multi line texts. It removes the new lines and insert all words in single list """

    def tokenize(self, text):
        return word_tokenize(text)


class NonAlphaNumCharTokenizer(object):
    """
    Replace the non alpha numeric character by space and tokenize the sentence by space.
    For example: the sentence 'hello world, org.eclipse.core.launcher.main.main' is tokenized to
    [hello, word , org, eclipse, core, launcher, main, main ].
    """
    REGEX = re.compile('[\W_]+', re.UNICODE)

    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()

    def tokenize(self, text):
        text = re.sub(NonAlphaNumericalChar.REGEX, ' ', text)

        return self.tokenizer.tokenize(text)


class HTMLSymbolFilter(Filter):

    def filter(self, token, sentence):
        return re.sub(r"((&quot)|(&gt)|(&lt)|(&amp)|(&nbsp)|(&copy)|(&reg))+", '', token)


class StopWordRemoval(Filter):

    def __init__(self, stopwords=stopwords.words('english')):
        self.stopwords = stopwords

    def filter(self, token, sentence):
        if token in self.stopWords:
            return ''

        return token


class NeutralQuotesFilter(Filter):
    """
    Transform neutral quotes("'`) to opening(``) or closing quotes('')
    """

    def __init__(self):
        super(NeutralQuotesFilter, self).__init__()
        self.__lastSentence = ""
        self.__isNextQuoteOpen = True

    def filter(self, token, sentence):
        if re.search(r"^[\"`']$", token):
            if self.__lastSentence == sentence:
                if self.__isNextQuoteOpen:
                    self.__isNextQuoteOpen = False
                    return "``"
                else:
                    self.__isNextQuoteOpen = True
                    return "''"
            else:
                self.__lastSentence = sentence
                self.__isNextQuoteOpen = False
                return "``"

        return token


class ModuleFunctionFilter(Filter):
    SYMBOL = '#MODULE_FUNCTION'

    def filter(self, token, sentence):
        a = str(token)
        # todo: this filter is matching with files (a.java, p.php, s.jsp, a.txt)
        pattern = r'^[a-zA-Z][a-zA-Z0-9]+(\.[a-zA-Z][a-zA-Z0-9]+)+(\(\))?$'
        token = re.sub(pattern, ModuleFunctionFilter.SYMBOL, token)

        return token

    def getSymbol(self):
        return self.SYMBOL


class NonAlphaNumericalChar(Filter):
    "Remove all non alpha numerical character"
    REGEX = re.compile('[\W_]+', re.UNICODE)

    def __init__(self, repl=' '):
        self.repl = repl

    def filter(self, token, sentence):
        token = re.sub(NonAlphaNumericalChar.REGEX, self.repl, token)

        return token


class UrlFilter(Filter):
    """
    Tokens starting with “www.”, “http.” or ending with “.org”, “.com” e ".net" are converted to a “#URL” symbol
    """

    SYMBOL = "#URL"

    def filter(self, token, sentence):
        a = str(token)
        token = re.sub(r"^((https?:\/\/)|(www\.))[^\s]+$", "#URL", token)
        # token = re.sub(r"^[^\s]+(\.com|\.net|\.org)\b([-a-zA-Z0-9@;:%_\+.~#?&//=]*)$", "#URL", token)
        token = re.sub(r"^[^\s]+(\.com|\.net|\.org)([/?]([-a-zA-Z0-9@;:%_\+.~#?&//=]*))?$", "#URL", token)

        return token

    def getSymbol(self):
        return self.SYMBOL


class RepeatedPunctuationFilter(Filter):
    """
    Repeated punctuations such as “!!!!” are collapsed into one.
    """

    def filter(self, token, sentence):
        token = re.sub(r"^([,:;><!?=_\\\/])\1{1,}$", '\\1', token)
        token = re.sub(r"^[.]{4,}$", "...", token)
        token = re.sub(r"^[.]{2,2}$", ".", token)
        token = re.sub(r"^[--]{3,}$", "--", token)

        return token


class StripPunctuactionFilter(Filter):
    REGEX = r"""(^[]\d!"#$%&'()*+,-.\/:;<=>?@[\^_`{|}~]+)|([]\d!"#$%&'()*+,-.\/:;<=>?@[\^_`{|}~]+$)"""

    def filter(self, token, sentence):
        token = re.sub(self.REGEX, '', token)

        return token


class DectectNotUsualWordFilter(Filter):
    puncSet = set(string.punctuation)

    def filter(self, token, sentence):
        # Remove sentences which 20% of characters are numbers or punctuations
        npt = 0

        if len(token) == 0:
            return token

        for c in token:
            if c.isnumeric() or c in self.puncSet:
                npt += 1

        if float(npt) / len(token) > 0.20:
            return ''

        return token


def loadFilters(filterNames):
    """
    Instance the filters using their names
    :param filterNames: class name
    :return:
    """
    filters = []
    module_ = importlib.import_module(Filter.__module__)

    for filterName in filterNames:
        filters.append(getattr(module_, filterName)())

    return filters


class PreprocessingCache(object):

    def __init__(self, folder=None, args=None):
        self.cache = {}

        # Cache needs folder and args to save it in the disk
        if folder is not None and args is not None:
            self.arguments = ' '.join(args)
            hashName = hashlib.sha256(self.arguments.encode()).hexdigest()
            filename = hashName + '.pkl'
            self.cacheFile = os.path.join(folder, filename)
            logger = logging.getLogger()
            self.persistent = True

            if os.path.isfile(self.cacheFile):
                logger.info("Cache: %s file was found" % self.cacheFile)
                logger.info("Loading cache")
                self.stored = True
                obj = pickle.load(open(self.cacheFile, 'rb'))

                if obj['args'] != self.arguments:
                    Exception('Hash colision!!!\nStr1: %s\nStr2: %s' % (obj['args'], self.arguments))

                self.cache = obj['cache']
            else:
                self.stored = False
                logger.info("Cache:%s file was not found" % self.cacheFile)
        else:
            self.persistent = False
            self.stored = False

    def add(self, bugId, ftrs):
        if bugId not in self.cache:
            self.cache[bugId] = ftrs
            # Cache was updated. We should save it again
            self.stored = False

    def get(self, bugId):
        return self.cache.get(bugId)

    def isStored(self):
        return self.stored

    def isPersistent(self):
        return self.persistent

    def save(self):
        logger = logging.getLogger()
        logger.info("Saving cache: %s" % self.cacheFile)

        pickle.dump({'args': self.arguments, 'cache': self.cache}, open(self.cacheFile, 'wb'))


class PreprocessorList(object):

    def __init__(self, preprocessors=None, cache=None):
        self.preprocessors = preprocessors if preprocessors else []
        self.cache = cache

    def __getitem__(self, item):
        return self.preprocessors[item]

    def append(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def extract(self, bugId):
        return [preprocessor.extract(bugId) for preprocessor in self.preprocessors]

    def __iter__(self):
        return iter(self.preprocessors)


class Preprocessor(object):

    def __init__(self, database, cache):
        self.database = database
        self.cache = cache if cache else PreprocessingCache()

    def _storeCache(self):
        if self.cache.isPersistent() and not self.cache.isStored():
            logger = logging.getLogger()

            logger.info("=>Preprocessing all reports.")

            for bug in self.database.bugList:
                bugId = bug['bug_id']
                self.cache.add(bugId, self._extract(bug))

            self.cache.save()

    def extract(self, bugId):
        ftrs = self.cache.get(bugId)

        if ftrs is None:
            ftrs = self._extract(self.database.getBug(bugId))
            self.cache.add(bugId, ftrs)

        return ftrs


class SABDEncoderPreprocessor(Preprocessor):
    maxSentenceSize = 350

    def __init__(self, lexicon, database, filters, tokenizer, paddingId, cache=None, field_padding_idx=0):
        super(SABDEncoderPreprocessor, self).__init__(database, cache)
        self.textProc = TextFieldPreprocessor(lexicon, filters, tokenizer)
        self.database = database
        self.paddingId = paddingId
        self.field_padding_idx = field_padding_idx
        logger = logging.getLogger()
        logger.info("Summary and Description: tokenizer class= %s" % tokenizer.__class__.__name__)

        self._storeCache()

    def _extract(self, bug):
        summary = bug['short_desc'].strip()
        field_ids = []

        if len(summary) == 0:
            summary = []
        else:
            summary = self.textProc.preprocess(summary)
            field_ids.extend(itertools.repeat(1, len(summary)))

        description = checkDesc(bug['description'])

        if len(description) == 0:
            description = []
        else:
            description = self.textProc.preprocess(description)[:self.maxSentenceSize]
            field_ids.extend(itertools.repeat(2, len(description)))

        if len(summary) + len(description) == 0:
            sum_desc = [self.paddingId]
            field_ids.append(self.field_padding_idx)
        else:
            sum_desc = summary + description

        return sum_desc, field_ids, None


class SummaryDescriptionPreprocessor(Preprocessor):
    maxSentenceSize = 350

    def __init__(self, lexicon, database, filters, tokenizer, paddingId, cache=None):
        super(SummaryDescriptionPreprocessor, self).__init__(database, cache)
        self.textProc = TextFieldPreprocessor(lexicon, filters, tokenizer)
        self.database = database
        logger = logging.getLogger()
        logger.info("Summary and Description: tokenizer class= %s" % tokenizer.__class__.__name__)

        self.paddingId = paddingId

        self._storeCache()

    def _extract(self, bug):
        summary = bug['short_desc'].strip()
        sum_desc = []

        if len(summary) > 0:
            sum_desc.extend(self.textProc.preprocess(summary))

        description = checkDesc(bug['description'])

        if len(description) > 0:
            sum_desc.extend(self.textProc.preprocess(description)[:self.maxSentenceSize])

        if len(sum_desc) == 0:
            sum_desc.append(self.paddingId)

        return sum_desc


class SummaryPreprocessor(Preprocessor):
    maxSentenceSize = 300

    def __init__(self, lexicon, database, filters, tokenizer, paddingId, cache=None):
        super(SummaryPreprocessor, self).__init__(database, cache)
        self.textProc = TextFieldPreprocessor(lexicon, filters, tokenizer)
        self.paddingId = paddingId

        logger = logging.getLogger()
        logger.info("Summary: tokenizer class= %s" % tokenizer.__class__.__name__)
        self._storeCache()

    def _extract(self, bug):
        summary = bug['short_desc']

        if len(summary.strip()) == 0:
            return [self.paddingId]

        return self.textProc.preprocess(summary)


class DescriptionPreprocessor(Preprocessor):
    maxSentenceSize = 300

    def __init__(self, lexicon, database, filters, tokenizer, paddindIdx, cache=None):
        super(DescriptionPreprocessor, self).__init__(database, cache)
        self.textProc = TextFieldPreprocessor(lexicon, filters, tokenizer)
        self.paddingIdx = paddindIdx

        logger = logging.getLogger()
        logger.info("Summary: tokenizer class= %s" % tokenizer.__class__.__name__)
        self._storeCache()

    def _extract(self, bug):
        description = checkDesc(bug['description'])

        if len(description) == 0:
            return [self.paddingIdx]

        return self.textProc.preprocess(description)[:self.maxSentenceSize]


class CategoricalPreprocessor(Preprocessor):

    def __init__(self, fieldTuples, database, cache=None):
        super(CategoricalPreprocessor, self).__init__(database, cache)
        self.fieldTuples = fieldTuples
        self._storeCache()

    def _extract(self, bug):
        ftrs = []

        for fieldName, lexicon, filters in self.fieldTuples:
            field = bug[fieldName]

            for filter in filters:
                field = filter.filter(field, field)

            ftrs.append(lexicon.put(field))

        return ftrs


class TextFieldPreprocessor(object):

    def __init__(self, lexicon, filters, tokenizer=None):
        self.lexicon = lexicon
        self.filters = filters
        self.tokenizer = tokenizer

    def _tokenize(self, field):
        if self.tokenizer:
            return self.tokenizer.tokenize(field)

        # We consider that the text is already tokenized
        return field.split(' ')

    def preprocess(self, field):
        words = self._tokenize(field)
        out = []

        for word in words:
            for filter in self.filters:
                word = filter.filter(word, field)

            if len(word) == 0:
                continue

            if self.lexicon is not None:
                word = self.lexicon.put(word)

            out.append(word)

        return out


class ClassicalPreprocessing:

    def __init__(self, tokenizer, stemmer, stopWords, filters=[]):
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopWords = stopWords
        self.filters = filters

    def preprocess(self, text):
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)

        cleanTokens = []

        for token in tokens:
            for fil in self.filters:
                token = fil.filter(token, text)

            if stopwords:
                if token in self.stopWords:
                    continue

            if self.stemmer:
                token = self.stemmer.stem(token)

            if len(token) > 0:
                cleanTokens.append(token)

        return cleanTokens


def concatenateSummaryAndDescription(bugReport):
    text = bugReport['short_desc'].strip()

    # If the summary doesn't have a dot at the end, so we concatenate summary and description using a dot and new line.
    # text += '.\n' if text[-1] != '.' else '\n'
    text += '\n'

    # Sometime the description is a empty list
    description = bugReport['description']
    if not isinstance(description, list):
        text += bugReport['description']

    return text


def cleanDescription(desc):
    # Remove class declaration
    cleanDesc = re.sub(CLASS_DEF_JAVA_REGEX, '', desc)

    # Remove if
    cleanDesc = re.sub(IF_JAVA_REGEX, '', cleanDesc)

    # Remove function, catch and some ifs
    cleanDesc = re.sub(FUNC_IF_DEF_JAVA_REGEX, '', cleanDesc)

    # Remove variablie
    cleanDesc = re.sub(OBJ_JAVA_REGEX, '', cleanDesc)

    # Remove time
    cleanDesc = re.sub(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}', '', cleanDesc)

    # Remove date
    cleanDesc = re.sub(r'[0-9]{1,4}([/-])[0-9]{1,2}\1[0-9]{2,4}', '', cleanDesc)

    # Remove repeated punctuation like ######
    cleanDesc = re.sub(r"([,:;><!?=_\\\/*-.,])\1{1,}", '\\1', cleanDesc)

    newdesc = ""
    puncSet = set(string.punctuation)

    for l in cleanDesc.split("\n"):
        # Remove sentence that have less 10 characters
        if len(l) < 10:
            continue

        # Remove the majority of stack traces, some code and too short texts. Remove sentence that has less 5 tokens.
        nTok = 0
        for t in re.split(r'\s', l):
            if len(t) > 0:
                nTok += 1

        if nTok < 5:
            continue

        # Remove sentences which 20% of characters are numbers or punctuations
        npt = 0
        for c in l:
            if c.isnumeric() or c in puncSet:
                npt += 1

        if float(npt) / len(l) > 0.20:
            continue

        newdesc += l + '\n'

    return newdesc


def softClean(text, rmPunc=False, sentTok=False, rmNumber=False, stop_words=False, stem=False, lower_case=False, rm_char=False):
    if lower_case:
        text = text.lower()

    cleanText = re.sub(RVM_REPEATED_PUNC, '\\1', text)

    # Remove time
    cleanText = re.sub(r'[0-9]{1,2}:[0-9]{2}:[0-9]{2}', '', cleanText)
    # Remove date
    cleanText = re.sub(r'[0-9]{1,4}([/-])[0-9]{1,2}\1[0-9]{2,4}', '', cleanText)

    if sentTok:
        cleanText = '\n'.join(sent_tokenize(text))

    tokens = word_tokenize(cleanText) if stop_words or stem else None

    if stem:
        stop_word_set = set(stopwords.words('english') + ["n't"]) if stop_words else set()
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        old_tokens = tokens
        tokens = []

        for token in old_tokens:
            nt = stemmer.stem(token)

            if len(nt) > 0 and nt not in stop_word_set:
                tokens.append(nt)

    elif stop_words:
        stop_word_set = set(stopwords.words('english') + ["n't"])
        tokens = list(filter(lambda token: token not in stop_word_set, tokens))

    if tokens is not None:
        cleanText = ' '.join(tokens)

    if rmPunc:
        cleanText = re.sub(SPLIT_PUNCT_REGEX, ' ', cleanText)
    else:
        cleanText = re.sub(SPLIT_PUNCT_REGEX, ' \\1 ', cleanText)

    if rmNumber:
        cleanText = re.sub(r'\b[0-9]+\b', '', cleanText)

    if rm_char:
        cleanText = re.sub(r'\b\S\b', '', cleanText)

    return cleanText