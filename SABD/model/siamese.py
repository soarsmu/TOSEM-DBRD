"""
In this module, there are NN that use siamese neural network and receive pairs.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear

from model.basic_module import MultilayerDense, meanVector


def computeListOutputSize(encoders):
    outputSize = 0

    for encoder in encoders:
        outputSize += encoder.getOutputSize()

    return outputSize


class WordMean(nn.Module):

    def __init__(self, wordEmbedding, updateEmbedding, hidden_size=0, standardization=False, dropout=0.0, batch_normalization=False):
        super(WordMean, self).__init__()

        if standardization:
            wordEmbedding.zscoreNormalization()

        self.embeddingSize = wordEmbedding.getEmbeddingSize()
        self.embedding = nn.Embedding(wordEmbedding.getNumberOfVectors(), self.embeddingSize, padding_idx=wordEmbedding.getPaddingIdx())
        self.embedding.weight.data.copy_(torch.from_numpy(wordEmbedding.getEmbeddingMatrix()))
        self.embedding.weight.requires_grad = updateEmbedding

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.batch_norm = nn.BatchNorm1d(self.embeddingSize) if batch_normalization else None
        self.hidden = Linear(self.embeddingSize, self.embeddingSize) if hidden_size > 0 else None

    def forward(self, x, initialHidden, lengths):
        x = self.embedding(x)

        if self.hidden:
            x = x + F.relu(self.hidden(x))

        x = meanVector(x, lengths)

        if self.batch_norm:
            x = self.batch_norm(x)

        if self.dropout:
            x = self.dropout(x)

        return x

    def getOutputSize(self):
        return self.embeddingSize


class CosinePairNN(nn.Module):

    def __init__(self, encoders):
        super(CosinePairNN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.encoders = ModuleList(encoders)
        self.logger.info("Cosine Pair NN")

    def encode(self, bugInput):
        x = [encoder(*input) for input, encoder in zip(bugInput, self.encoders)]
        x = torch.cat(x, 1)

        return x

    def forward(self, bug1, bug2):
        """
        :param inputs: (batch, seq_len) tensor containing word indexes for each example
        :return: (batch, num_classes) tensor containing scores for each class
        """
        bugEmb1 = self.encode(bug1)
        bugEmb2 = self.encode(bug2)

        return self.similarity(bugEmb1, bugEmb2)

    def similarity(self, bugEmb1, bugEmb2):
        return F.cosine_similarity(bugEmb1, bugEmb2)


class CosineTripletNN(nn.Module):

    def __init__(self, encoders, dropout=0.0):
        super(CosineTripletNN, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.encoders = ModuleList(encoders)
        self.logger.info("Cosine Triplet NN")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def encode(self, bugInput):
        x = [encoder(*input) for input, encoder in zip(bugInput, self.encoders)]
        x = torch.cat(x, 1)

        if self.dropout:
            x = self.dropout(x)

        return x

    def forward(self, anchor, pos, neg):
        """
        :param inputs: (batch, seq_len) tensor containing word indexes for each example
        :return: (batch, num_classes) tensor containing scores for each class
        """
        anchorEmb = self.encode(anchor)
        posEmb = self.encode(pos)
        negEmb = self.encode(neg)

        return self.similarity(anchorEmb, posEmb), self.similarity(anchorEmb, negEmb)

    def similarity(self, bugEmb1, bugEmb2):
        return F.cosine_similarity(bugEmb1, bugEmb2)


class ProbabilityPairNN(nn.Module):
    """
    """

    def __init__(self, encoders, withoutBugEmbedding=False, hiddenLayerSizes=[100], batchNormalization=True, dropout=0.0):
        super(ProbabilityPairNN, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.encoders = ModuleList(encoders)
        encOutSize = computeListOutputSize(encoders)

        self.logger.info("%sUsing raw embeddings" % ("Not " if withoutBugEmbedding else ""))
        hiddenInput = 2 * encOutSize if withoutBugEmbedding else 4 * encOutSize

        self.withoutBugEmbedding = withoutBugEmbedding

        self.logger.info(
            "Probability Pair NN: without_raw_bug={}, batch_normalization={}".format(self.withoutBugEmbedding,
            batchNormalization))

        seq = []
        last = hiddenInput
        for currentSize in hiddenLayerSizes:
            seq.append(nn.Linear(last, currentSize))

            if batchNormalization:
                seq.append(nn.BatchNorm1d(currentSize))

            seq.append(nn.ReLU())

            if dropout > 0.0:
                seq.append(nn.Dropout(dropout))

            self.logger.info("==> Create Hidden Layer (%d,%d) in the classifier" % (last, currentSize))
            last = currentSize

        seq.append(nn.Linear(last, 2))

        self.sequential = Sequential(*seq)

    def encode(self, bugInput):
        x = [encoder(*input) for input, encoder in zip(bugInput, self.encoders)]
        x = torch.cat(x, 1)

        return x

    def forward(self, bug1, bug2):
        """
        :param inputs: (batch, seq_len) tensor containing word indexes for each example
        :return: (batch, num_classes) tensor containing scores for each class
        """
        bugEmb1 = self.encode(bug1)
        bugEmb2 = self.encode(bug2)

        return self.similarity(bugEmb1, bugEmb2)

    def similarity(self, bugEmb1, bugEmb2):
        hiddenIn = [torch.pow(bugEmb2 - bugEmb1, 2), bugEmb2 * bugEmb1]

        if not self.withoutBugEmbedding:
            hiddenIn.append(bugEmb1)
            hiddenIn.append(bugEmb2)

        x = torch.cat(hiddenIn, 1)
        x = self.sequential(x)

        return F.log_softmax(x, dim=1)


class CategoricalEncoder(nn.Module):
    """
    Encode the categorical information into a vector.
    """

    def __init__(self, lexicons, embeddingSize, hiddenSizes, activationFunc=F.tanh, batchNormalization=False, applyBatchLastLayer=True, dropoutLastLayer=0.0, layerNorm=False):
        super(CategoricalEncoder, self).__init__()

        logging.getLogger(__name__).info("Categorical Encoder: emb_size={}".format(embeddingSize))

        self.embeddingSize = embeddingSize
        self.embeddings = ModuleList([nn.Embedding(lex.getLen(), self.embeddingSize) for lex in lexicons])
        self.dense = MultilayerDense(len(lexicons) * embeddingSize, hiddenSizes, activationFunc, batchNormalization, applyBatchLastLayer, dropoutLastLayer, layerNorm)

    def forward(self, x):
        embList = []

        for em, _in in zip(self.embeddings, x):
            embList.append(em(_in))

        x = torch.cat(embList, 1)
        return self.dense(x)

    def getOutputSize(self):
        return self.dense.getOutputSize()
