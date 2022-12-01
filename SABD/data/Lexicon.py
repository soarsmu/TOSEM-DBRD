#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""
import codecs
import logging

import numpy


class Lexicon(object):
    """
    Represents a lexicon of words.
    Each added word in this lexicon will be represented by a integer.
    This class has a variable '__readOnly' that control if the lexicon will or not insert new words.
    If a word in new to lexicon and '__readOnly' is true, so this lexicon will return a index which
        is related with a unknown word.
    """

    def __init__(self, unknownSymbol="UUUNKKK", name=None):
        """
        :param unknownSymbol: the string which represents all unknown words.
            If this parameter is none, so a warning message will be showed and a default unknown symbol is generated.
        :param name: the name of the object. This parameter will be used when you save this object.
            If this parameter is None, so it won't be possible to save it.
        """
        self.__lexicon = []
        self.__lexiconDict = {}
        self.__readOnly = False
        self.__name = name
        self.__log = logging.getLogger(__name__)

        # Keep how many times a word was inserted in the lexicon
        self.__countInsWord = []

        if unknownSymbol is None:
            name = name if name is not None else ""
            self.__log.warning("The unknown symbol of the lexicon '" + name + "' wasn't defined.")

            self.unknown_index = -1
        else:
            self.unknown_index = self.put(unknownSymbol, False)

    def isReadOnly(self):
        """
        :return: return if lexicon is adding new words
        """
        return self.__readOnly

    def getLexiconList(self):
        """
        :return: return a list of all words of the lexicon. The position of the word in the list is the idx of this word
            in the lexicon
        """
        return self.__lexicon

    def getLexiconDict(self):
        """
        :return: return a dictionary which contains the integers which represent each word.
        """
        return self.__lexiconDict

    def getLen(self):
        """
        Return the number of words in the lexicon.
        """
        return len(self.__lexicon)

    def put(self, word, isToCount=True):
        """
        Include a new word in the lexicon and return its index. If the word is
        already in the lexicon, then just return its index.
        If a word in new to lexicon and '__readOnly' is true, so this lexicon will return a index which
        is related with all unknown words.
        """

        idx = self.__lexiconDict.get(word)

        if idx is None:
            if self.isReadOnly():
                return self.getUnknownIndex()

            # Insert a unseen word in the lexicon.
            idx = len(self.__lexicon)
            self.__lexicon.append(word)
            self.__lexiconDict[word] = idx

            self.__countInsWord.append(0)

        if not self.isReadOnly() and isToCount:
            # Count how many times a word was insert in the lexicon
            # Stop to count when the lexicon is only read
            self.__countInsWord[idx] += 1

        return idx

    def getLexicon(self, index):
        """
        Return the word in the lexicon that is stored in the given index.
        """
        return self.__lexicon[index]

    def getLexiconIndex(self, word):
        """
        Return the index of the given word. If the word is not in the lexicon,
            so returns the unknown index.
        """
        return self.__lexiconDict.get(word, self.unknown_index)

    def setUnknown(self, unknownSymbol):
        self.unknown_index = self.getLexiconIndex(unknownSymbol)

    def getUnknownIndex(self):
        return self.unknown_index

    def isUnknownIndex(self, index):
        return index == self.unknown_index

    def isUnknownSymbolDefined(self):
        return self.unknown_index != -1

    def exist(self, word):
        return not self.isUnknownIndex(self.getLexiconIndex(word))

    def stopAdd(self):
        """
        Tell the class to stop adding new words
        :return:
        """
        self.__readOnly = True

    def prune(self, minCount):
        """
        Remove the words that weren't inserted lesser than minCount.
        The lexicon can't be pruned when it's read only.
        :param minCount: Minimum number of times a word most have been inserted to don't be removed
        :return:
        """
        newLexicon = []
        newLexiconDict = {}
        newCountInsWord = []

        if self.isReadOnly():
            self.__log.warning("You can't prune a read only lexicon")
            return

        # The removed words  from dictionary will be treated as unknown and
        # yours frequencies will be added to unknown
        countNewUnknown = 0

        for idx, nm in enumerate(self.__countInsWord):
            word = self.__lexicon[idx]

            if nm >= minCount or idx is self.getUnknownIndex():
                newIdx = len(newLexicon)
                newLexicon.append(word)
                newLexiconDict[word] = newIdx
                newCountInsWord.append(nm)
            else:
                countNewUnknown += nm

        if self.isUnknownSymbolDefined():
            newCountInsWord[self.getUnknownIndex()] = countNewUnknown

        self.__log.info("Number of words pruned using minCount=%d:  %d of %d" % (
        minCount, self.getLen() - len(newLexicon), self.getLen()))

        self.__lexicon = newLexicon
        self.__lexiconDict = newLexiconDict
        self.__countInsWord = newCountInsWord

    def getFrequencyOfAllWords(self):
        '''
        It returns a vector which contains the frequency of all words.
        The words are encoded by integers and each dimension of this vector is the frequency
        of a specific word.
        :return a numpy vector of integers
        '''

        if not self.isReadOnly():
            self.__log.warning(
                "The word frequencies can change, because the lexicon is not read only ")

        return numpy.asarray(self.__countInsWord)

    @staticmethod
    def fromTextFile(filePath, hasUnknowSymbol, lexiconName=None):
        """
        Create lexicon object from a text file
        :param filePath: path of the file with the words
        :param hasUnknowSymbol: if this parameter is true, so the first word of the file will be consider as the unknown
            symbol. However, if this parameter is false, so the unknown symbol will be undefined.
        :param lexiconName: name of lexicon
        :return: Lexicon
        """
        f = codecs.open(filePath, "r", encoding="utf-8")

        if hasUnknowSymbol:
            unknownSym = f.readline().strip("\n")
        else:
            unknownSym = None

        lexicon = Lexicon(unknownSym, lexiconName)

        for l in f:
            lexicon.put(l.rstrip("\n"))

        lexicon.stopAdd()

        return lexicon

    @staticmethod
    def fromList(labels, hasUnknowSymbol, lexiconName=None):
        """
        Create lexicon object from a list of labels.
        :param labels: list of labels
        :param hasUnknowSymbol: if this parameter is true, so the first word of the file will be consider as the unknown
            symbol. However, if this parameter is false, so the unknown symbol will be undefined.
        :param lexiconName: name of lexicon
        :return: Lexicon
        """
        idx = 0
        if hasUnknowSymbol:
            unknownSym = labels[idx]
            idx += 1
        else:
            unknownSym = None

        lexicon = Lexicon(unknownSym, lexiconName)

        for l in labels[idx:]:
            lexicon.put(l)

        lexicon.stopAdd()

        return lexicon

    def save(self, filePath):
        """
        Save a lexicon in a text file
        :return: None
        """
        f = codecs.open(filePath, "w", encoding="utf-8")

        for word in self.__lexicon:
            f.write(word)
            f.write("\n")

    def getName(self):
        return self.__name

    def getAttributes(self):
        return {
            "lexicon": [word.encode("unicode_escape") for word in self.__lexicon],
            "unknownIndex": self.unknown_index
        }

    @staticmethod
    def fromPersistentManager(persistentManager, name):
        """
        Return a lexicon from a PersistentManager
        :type persistentManager: persistence.PersistentManager.PersistentManager
        :param persistentManager: this object is a bridge between the database and PersistentObject
        :type name: basestring
        :param name: name of the object in database
        :return: Lexicon
        """
        newLexicon = Lexicon(name=name)
        persistentManager.load(newLexicon)

        return newLexicon

    def load(self, attributes):
        lexicon = attributes["lexicon"]
        # The only way that I found to tranform a Dataset to integer
        self.unknown_index = numpy.int(numpy.asarray(attributes["unknownIndex"]))

        if self.unknown_index == -1:
            name = self.__name if self.__name is not None else ""
            self.__log.warning("The unknown symbol of the lexicon '" + name + "' wasn't defined.")

        self.__lexicon = []
        self.__lexiconDict = {}
        self.__readOnly = False

        for word in lexicon:
            word = word.decode("unicode_escape")
            self.put(word)

        self.__readOnly = True

    def save(self, filePath):
        """
        Save a lexicon in a text file
        :return: None
        """
        f = codecs.open(filePath, "w", encoding="utf-8")

        for word in self.__lexicon:
            f.write(word)
            f.write("\n")

