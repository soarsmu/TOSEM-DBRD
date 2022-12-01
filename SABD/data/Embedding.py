#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes and functions that operate on distributed representations
"""

import codecs
import logging
import numpy as np

import numpy
from data.Lexicon import Lexicon


def generateVector(num_features, min_value=-0.1, max_value=0.1):
    return (max_value * 2) * numpy.random.random_sample(num_features) + min_value


#####################################################################
# Embedding classes
####################################################################

class Embedding(object):
    """
    Represents an object distributed representation.
    This class has a matrix with all vectors and lexicon.
    """

    def __init__(self, lexicon, vectors=None, embeddingSize=None, paddingIdx=None):
        """
        Creates a embedding object from lexicon and vectors.
        If vectors is none, so each word in the lexicon will be represented by a random vector with embeddingSize dimensions.
        :type lexicon: data.Lexicon.Lexicon
        :params lexicon: a Lexicon object
        :type vectors: [[int]] | numpy.array | None
        :params vectors: embedding list
        :params embeddingSize: the number of dimensions of vectors. This only will be used when the vectors is none
        """
        self.__lexicon = lexicon
        self.__paddingIdx = paddingIdx

        if vectors is None:
            numVectors = lexicon.getLen()
            vectors = []

            for _ in range(numVectors):
                vec = generateVector(embeddingSize)
                vectors.append(vec)

        self.__vectors = np.asarray(vectors)
        self.__embeddingSize = self.__vectors.shape[1]

        if lexicon.getLen() != self.__vectors.shape[0]:
            raise Exception("The number of embeddings is different of lexicon size ")

        lexicon.stopAdd()

        if not lexicon.isReadOnly():
            raise Exception(
                "It's possible to insert in the lexicon. Please, transform the lexicon to only read.")

    def getPaddingIdx(self):
        return self.__paddingIdx

    def exist(self, obj):
        return self.__lexicon.exist(obj)

    def getLexiconIndex(self, obj):
        return self.__lexicon.getLexiconIndex(obj)

    def getEmbeddingByIndex(self, idx):
        return self.__vectors[idx]

    def getEmbedding(self, obj):
        idx = self.__lexicon.getLexiconIndex(obj)
        return self.getEmbeddingByIndex(idx)

    def getEmbeddingMatrix(self):
        return self.__vectors

    def getNumberOfVectors(self):
        return len(self.__vectors)

    def getEmbeddingSize(self):
        return self.__embeddingSize

    def getLexicon(self):
        """
        :return data_operation.lexicon.Lexicon
        """
        return self.__lexicon

    def zscoreNormalization(self, norm_coef=1.0):
        """
        Normalize all the embeddings using the following equation:
        x = (x − mean(x)) / stddev(x)
        :return: None
        """
        mean = np.mean(self.__vectors, axis=0)
        std = np.std(self.__vectors, axis=0)

        self.__vectors = ((self.__vectors - mean) / std)

    def minMaxNormalization(self, norm_coef=1.0):
        """
        Normalize all the embeddings to a range [0,1].
        x = (x − min(x)) / (max(x) − min(x))
        :return:None
        """
        raise Exception("Wrong")
        self.__vectors = np.asarray(self.__vectors)

        self.__vectors -= np.min(self.__vectors, axis=0)
        self.__vectors *= (norm_coef / np.ptp(self.__vectors, axis=0))

    def meanNormalization(self, norm_coef=1.0):
        """
        Normalize all the embeddings to a range [-1,1].
        x = (x − mean(x)) / (max(x) − min(x))
        :return:None
        """
        self.__vectors = np.asarray(self.__vectors)
        raise Exception("Wrong")
        self.__vectors -= np.mean(self.__vectors, axis=0)
        self.__vectors *= (norm_coef / np.ptp(self.__vectors, axis=0))

    @staticmethod
    def fromFile(file, unknownSymbol, lexiconName=None, hasHeader=True, paddingSym=None):
        """
        Creates  a lexicon and a embedding from word2vec file.
        :param file: path of file
        :param unknownSymbol: the string that represents the unknown words.
        :return: (data.Lexicon.Lexicon, Embedding)
        """
        log = logging.getLogger(__name__)
        fVec = codecs.open(file, 'r', 'utf-8')

        # Read the number of words in the dictionary and the embedding size
        if hasHeader:
            nmWords, embeddingSizeStr = fVec.readline().strip().split(" ")
            embeddingSize = int(embeddingSizeStr)
        else:
            embeddingSize = None

        lexicon = Lexicon(unknownSymbol, lexiconName)
        # The empty array represents the array of unknown
        # At end, this array will be replaced by one array that exist in the  w2vFile or a random array.
        vectors = [[]]
        nmEmptyWords = 0

        for line in fVec:
            splitLine = line.rstrip().split(u' ')
            word = splitLine[0]

            if len(word) == 0:
                log.warning("Insert in the embedding a empty string. This embeddings will be thrown out.")
                nmEmptyWords += 1
                continue

            vec = [float(num) for num in splitLine[1:]]

            if word == unknownSymbol:
                if len(vectors[0]) != 0:
                    raise Exception("A unknown symbol was already inserted.")

                vectors[0] = vec
            else:
                lexicon.put(word)
                vectors.append(vec)

        expected_size = lexicon.getLen() - 1 + nmEmptyWords

        if len(vectors[0]) == 0:
            if embeddingSize is None:
                embeddingSize = len(vectors[-1])

            vectors[0] = generateVector(embeddingSize)
            expected_size += 1

        if hasHeader:
            if int(nmWords) != expected_size:
                raise Exception("The size of lexicon is different of number of vectors")

        if paddingSym is None:
            paddingIdx = None
        else:
            if not lexicon.exist(paddingSym):
                paddingIdx = lexicon.put(paddingSym)
                vectors.append([0.0] * embeddingSize)
            else:
                paddingIdx = lexicon.getLexiconIndex(paddingSym)

        fVec.close()

        return lexicon, Embedding(lexicon, vectors, paddingIdx=paddingIdx)
