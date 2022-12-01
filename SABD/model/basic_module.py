import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Module, Dropout, LayerNorm, BatchNorm1d

from data.Embedding import Embedding
from model.attention import SelfAttention


def sortInput(x, initialHidden, lengths, isLSTM):
    # Sort by sentence length because of pack_padded_sequence
    lengths, sortedIdxs = torch.sort(lengths, descending=True)

    if initialHidden is not None:
        if isLSTM:
            initialHidden = (initialHidden[0][:, sortedIdxs, :], initialHidden[1][:, sortedIdxs, :])
        else:
            initialHidden = initialHidden[:, sortedIdxs, :]

    return x[sortedIdxs], initialHidden, sortedIdxs, lengths


def undoSort(x, lengths, sortedIdxs):
    """
    Come back to the original sentence positions
    Solution Multiple inputs:
        https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106/3
    """''

    return torch.zeros_like(x).scatter_(0, sortedIdxs.unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2]), x)


# Pooling methods to create a fixed size - RNN
def meanVector(x, lengths, keep_dim=False):
    if keep_dim:
        # Calculate Mean
        return torch.sum(x, 1, keepdim=True) / lengths.view(-1, 1, 1).float()
    else:
        return torch.sum(x, 1) / torch.unsqueeze(lengths, 1).float()


def maxVector(x, lengths):
    # Calculate Max
    maxLength = x.size()[1]

    """
    The pads, whixh are zero vectors, can have the biggest values of the vectors and 
    their number can be returned by the max-poling. 
    We masking these values by adding min-value to these padding.
    """
    minValue = torch.min(x, 1, True)[0]
    boolMask = torch.arange(0, maxLength, device=lengths.device).unsqueeze(1).unsqueeze(0).expand(
        x.size()) >= lengths.unsqueeze(1).unsqueeze(
        1).expand(x.size())
    mask = minValue * boolMask.float()

    maskedX = x + mask

    return torch.max(maskedX, 1)[0]


def createEmbeddingLayer(embeddingObject, updateEmbedding):
    embedding = nn.Embedding(embeddingObject.getNumberOfVectors(), embeddingObject.getEmbeddingSize(), padding_idx=embeddingObject.getPaddingIdx())
    embedding.weight.data.copy_(torch.from_numpy(embeddingObject.getEmbeddingMatrix()))

    embedding.weight.requires_grad = updateEmbedding

    inputSize = embeddingObject.getEmbeddingSize()

    return embedding, inputSize


def lastVector(x, lengths):
    # Get last non-pad vector of the RNN
    return x[torch.arange(lengths.size()[0]), (lengths - 1), :]


# Basic NN
class ResidualLayer(nn.Module):

    def __init__(self, embedding, activation_fn):
        super(ResidualLayer, self).__init__()
        self.embedding = embedding
        self.hidden_layer = nn.Linear(embedding.embedding_dim, embedding.embedding_dim)
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.embedding(input)
        x2 = self.activation_fn(self.hidden_layer(x))

        return x + x2

    def getOutputSize(self):
        return self.embedding.embedding_dim


class RNNEncoder(nn.Module):
    """
    Encode the summary information into a vector using a LSTM.
    """

    def __init__(self, rnnType, embeddingObject, hiddenSize=100, numLayers=1, bidirectional=False, updateEmbedding=True, dropout=0, isLengthVariable=True):
        super(RNNEncoder, self).__init__()

        self.numLayers = numLayers
        self.hiddenSize = hiddenSize
        self.bidirectional = bidirectional
        self.isLengthVariable = isLengthVariable

        # Copy pre-trained embedding to the layer
        if isinstance(embeddingObject, Embedding):
            self.embedding = nn.Embedding(embeddingObject.getNumberOfVectors(), embeddingObject.getEmbeddingSize(), padding_idx=embeddingObject.getPaddingIdx())

            self.embedding.weight.data.copy_(torch.from_numpy(embeddingObject.getEmbeddingMatrix()))

            self.embedding.weight.requires_grad = updateEmbedding

            inputSize = embeddingObject.getEmbeddingSize()
        elif isinstance(embeddingObject, int):
            inputSize = embeddingObject
            self.embedding = None
        else:
            self.embedding = embeddingObject
            self.embedding.weight.requires_grad = updateEmbedding

            inputSize = self.embedding.embedding_dim

        logging.getLogger(__name__).info(
            "RNN Encoder: type={}, hiddenSize={}, numLayers={}, update_embedding={}, bidirectional={}, dropout{}".format(
                rnnType, self.hiddenSize, numLayers, updateEmbedding, bidirectional, dropout))

        self.isLSTM = rnnType == 'lstm'
        self.dropout = dropout if numLayers > 1 else 0.0

        if rnnType == 'lstm':
            self.rnnEncoder = nn.LSTM(inputSize, self.hiddenSize, self.numLayers, batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        elif rnnType == 'gru':
            self.rnnEncoder = nn.GRU(inputSize, self.hiddenSize, self.numLayers, batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        else:
            raise Exception('rnn type {} is not valid.'.format(rnnType))

    def forward(self, x, initialHidden, lengths):
        """
        The batches have sequences of different sizes and, to pass them to LSTM, we have to sort our batch by
        sequence lengths and use pack_padded_sequence and pad_packed_sequence functions.
        After generate the vector of each sequence, we come back the original order of the sequences in the batch.
        """
        # Forward propagation
        input = self.embedding(x) if self.embedding else x

        if self.isLengthVariable:
            input = nn.utils.rnn.pack_padded_sequence(input, lengths.cpu(), batch_first=True)

        output, hidden = self.rnnEncoder(input, initialHidden)

        if self.isLengthVariable:
            output, lengthList = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden

    def getOutputSize(self):
        bidirectionalFactor = 2 if self.bidirectional else 1
        return self.hiddenSize * bidirectionalFactor


class SortedRNNEncoder(nn.Module):

    def __init__(self, rnnType, embeddingObject, hiddenSize=100, numLayers=1, bidirectional=False, updateEmbedding=True, dropout=0):
        super(SortedRNNEncoder, self).__init__()
        self.rnnEncoder = RNNEncoder(rnnType, embeddingObject, hiddenSize, numLayers, bidirectional, updateEmbedding, dropout, True)

    def forward(self, x, initialHidden, lengths):
        sortedX, initialHidden, sortedIdxs, lengths = sortInput(x, initialHidden, lengths, self.rnnEncoder.isLSTM)
        sortedX, hidden = self.rnnEncoder(sortedX, initialHidden, lengths)

        unsortedX = undoSort(sortedX, lengths, sortedIdxs)

        return unsortedX, None

    def getOutputSize(self):
        return self.rnnEncoder.getOutputSize()


class CNN(nn.Module):
    """
        Implementing a Convolutional Neural Networks for Sentence Classification
        """

    def __init__(self, inputSize, windowSizes, nFilters, activationFunc=F.relu, batchNormalization=False, dropout=0.0):
        super(CNN, self).__init__()

        self.nFilters = nFilters
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        logging.getLogger(__name__).info(
            "Text CNN: nfilters={}, window sizes={}, inputSize={}, batch_norm={}, dropout={}".format(
                self.nFilters,
                windowSizes,
                inputSize,
                batchNormalization,
                dropout))

        self.convs = nn.ModuleList([nn.Conv1d(inputSize, nFilters, k) for k in windowSizes])
        self.activationFunc = activationFunc
        self.bns = nn.ModuleList(nn.BatchNorm1d(nFilters) if batchNormalization else None for _ in windowSizes)

    def conv_and_max_pool(self, x, conv, bn):
        """
        Convolution and max pooling operation.

        :param x: (batch, seq_len, emb_dim) tensor
        :param conv: Conv1d module
        :return: (batch, num_filters) tensor
        """
        # x.shape = (batch, seq_len, emb_dim)
        # Permute second and third dimensions because Conv1d expects seq_len in the last dimension.
        x = x.permute(0, 2, 1)
        # x.shape = (batch, emb_dim, seq_len)
        x = conv(x)

        if bn:
            x = bn(x)

        # x.shape = (batch, num_filters, ~seq_len)
        x = x.permute(0, 2, 1)
        # x.shape = (batch, seq_len, num_filters)
        x = x.max(1)[0]
        # x.shape = (batch, num_filters)

        return self.activationFunc(x)

    def forward(self, x):
        """
        :param inputs: (batch, seq_len) tensor containing word indexes for each example
        :return: (batch, num_classes) tensor containing scores for each class
        """
        # x.shape = (batch, seq_len, emb_dim)
        x = [self.conv_and_max_pool(x, conv, bn) for conv, bn in zip(self.convs, self.bns)]

        # x is a list with len(kernel_sizes) items with shape = (batch, num_filters)
        x = torch.cat(x, 1)

        if self.dropout:
            x = self.dropout(x)

        return x

    def getOutputSize(self):
        return self.nFilters * len(self.convs)


class TextCNN(nn.Module):
    """
    Implementing a Convolutional Neural Networks for Sentence Classification
    """

    def __init__(self, windowSizes, nFilters, wordEmbedding, updateEmbedding, activationFunc=F.relu, batchNormalization=False, dropout=0.0):
        super(TextCNN, self).__init__()

        embeddingSize = wordEmbedding.getEmbeddingSize()
        self.embedding = nn.Embedding(wordEmbedding.getNumberOfVectors(), embeddingSize, padding_idx=wordEmbedding.getPaddingIdx())
        self.embedding.weight.data.copy_(torch.from_numpy(wordEmbedding.getEmbeddingMatrix()))
        self.embedding.weight.requires_grad = updateEmbedding
        self.nFilters = nFilters
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        logging.getLogger(__name__).info(
            "Text CNN: nfilters={}, window sizes={}, update_embedding={}, batch_norm={}, dropout={}".format(
                self.nFilters,
                windowSizes,
                updateEmbedding,
                batchNormalization,
                dropout))

        self.convs = nn.ModuleList([nn.Conv1d(embeddingSize, nFilters, k) for k in windowSizes])
        self.activationFunc = activationFunc
        self.bns = nn.ModuleList(nn.BatchNorm1d(nFilters) if batchNormalization else None for _ in windowSizes)

    def conv_and_max_pool(self, x, conv, bn):
        """
        Convolution and max pooling operation.

        :param x: (batch, seq_len, emb_dim) tensor
        :param conv: Conv1d module
        :return: (batch, num_filters) tensor
        """
        # x.shape = (batch, seq_len, emb_dim)
        # Permute second and third dimensions because Conv1d expects seq_len in the last dimension.
        x = x.permute(0, 2, 1)
        # x.shape = (batch, emb_dim, seq_len)
        x = conv(x)

        if bn:
            x = bn(x)

        # x.shape = (batch, num_filters, ~seq_len)
        x = x.permute(0, 2, 1)
        # x.shape = (batch, seq_len, num_filters)
        x = x.max(1)[0]
        # x.shape = (batch, num_filters)

        return self.activationFunc(x)

    def forward(self, inputs):
        """
        :param inputs: (batch, seq_len) tensor containing word indexes for each example
        :return: (batch, num_classes) tensor containing scores for each class
        """
        # input.shape = (batch, seq_len)
        x = self.embedding(inputs)

        # x.shape = (batch, seq_len, emb_dim)
        x = [self.conv_and_max_pool(x, conv, bn) for conv, bn in zip(self.convs, self.bns)]

        # x is a list with len(kernel_sizes) items with shape = (batch, num_filters)
        x = torch.cat(x, 1)

        if self.dropout:
            x = self.dropout(x)

        return x

    def getOutputSize(self):
        return self.nFilters * len(self.convs)


class MultilayerDense(Module):

    def __init__(self, previousLayer, hiddenSizes, activationFunc, batchNormalization=False, applyBatchLastLayer=True, dropoutLastLayer=0.0, layerNorm=False):
        super(MultilayerDense, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Multilayer NN: activation={}, batch_normalization={}, dropout_last_layer: {}".format(
                activationFunc.__name__, batchNormalization, dropoutLastLayer))
        seq = []
        if isinstance(previousLayer, int):
            last = previousLayer
        else:
            last = previousLayer.getOutputSize()
            seq.append(previousLayer)

        if layerNorm:
            norm_class = LayerNorm
        elif batchNormalization:
            norm_class = BatchNorm1d
        else:
            norm_class = None

        self.outputSize = hiddenSizes[-1]
        for idx, currentSize in enumerate(hiddenSizes):
            seq.append(nn.Linear(last, currentSize))

            if norm_class is not None and not (idx + 1 == len(hiddenSizes) and not applyBatchLastLayer):
                seq.append(norm_class(currentSize))

            seq.append(activationFunc())
            self.logger.info("==> Create Hidden Layer (%d,%d) in the encoder" % (last, currentSize))
            last = currentSize

        if dropoutLastLayer > 0.0:
            seq.append(Dropout(dropoutLastLayer))

        self.sequential = Sequential(*seq)

    def forward(self, x):
        return self.sequential(x)

    def getOutputSize(self):
        return self.outputSize


def mean_max(outputs, lengths):
    return torch.cat([meanVector(outputs, lengths), maxVector(outputs, lengths)], 1)


class Dense_Self_Attention(nn.Module):

    def __init__(self, wordEmbedding, hidden_size, self_att_hidden, n_hops, paddingId, updateEmbedding, dropout=None):
        super(Dense_Self_Attention, self).__init__()

        embeddingSize = wordEmbedding.getEmbeddingSize()
        self.embedding = nn.Embedding(wordEmbedding.getNumberOfVectors(), embeddingSize, padding_idx=wordEmbedding.getPaddingIdx())
        self.embedding.weight.data.copy_(torch.from_numpy(wordEmbedding.getEmbeddingMatrix()))
        self.embedding.weight.requires_grad = updateEmbedding

        self.dense = nn.Linear(embeddingSize, embeddingSize) if hidden_size else None

        self.self_attention = SelfAttention(embeddingSize, self_att_hidden, n_hops)
        self.paddingId = paddingId
        self.output_size = self.self_attention.getOutputSize()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        mask = (x != self.paddingId).float()

        x = self.embedding(x)

        if self.dense is not None:
            res = F.relu(self.dense(x)) * mask.unsqueeze(2)
            x = x + res

        x, att = self.self_attention(x, mask)
        x = x.view(x.size(0), self.output_size)

        if self.dropout:
            self.dropout(x)

        return x

    def getOutputSize(self):
        return self.self_attention.getOutputSize()


class RNN_Self_Attention(nn.Module):

    def __init__(self, rnn, self_attention, paddingId, dropout=None):
        super(RNN_Self_Attention, self).__init__()
        self.rnn = rnn
        self.paddingId = paddingId
        self.self_attention = self_attention
        self.output_size = self_attention.getOutputSize()
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x, initialHidden, lengths, dropout=None):
        mask = (x != self.paddingId).float()

        outputs, hiddens = self.rnn(x, initialHidden, lengths)

        x, att = self.self_attention(outputs, mask)

        x = x.view(outputs.size(0), self.output_size)

        if self.dropout:
            self.dropout(x)

        return x

    def getOutputSize(self):
        return self.self_attention.getOutputSize()


class RNNFixedOuput(nn.Module):

    def __init__(self, rnn, fixedSizeMethod, dropout=None):
        super(RNNFixedOuput, self).__init__()

        if fixedSizeMethod == 'last':
            self.fixedSizeFunc = lastVector
        elif fixedSizeMethod == 'mean':
            self.fixedSizeFunc = meanVector
        elif fixedSizeMethod == 'max':
            self.fixedSizeFunc = maxVector
        elif fixedSizeMethod == 'mean+max':
            self.fixedSizeFunc = mean_max

        self.rnn = rnn
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x, initialHidden, lengths):
        outputs, hiddens = self.rnn(x, initialHidden, lengths)

        x = self.fixedSizeFunc(outputs, lengths)

        if self.dropout:
            self.dropout(x)

        return x

    def getOutputSize(self):
        return self.rnn.getOutputSize() * 2 if self.fixedSizeFunc == mean_max else self.rnn.getOutputSize()
