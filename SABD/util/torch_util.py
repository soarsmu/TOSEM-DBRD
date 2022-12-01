import math
from argparse import ArgumentError

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import tanh, sigmoid

def thresholded_output_transform(x):
    if len(x) == 3:
        loss, y_pred, y = x
    else:
        y_pred, y = x

    y_pred = y_pred.round()

    return y_pred, y

def padSequences(sequences, value, dtype='int32', minSize=-1, maxSize=math.inf):
    """
    Pad sequences to same length.

    :param sequences: a list of list
    :param value: value used to pad
    :param dtype:
    :param minSize: guarantee that all inputs have at least a minimum length.
    :return:
    """
    maxLength = minSize

    for seq in sequences:
        if len(seq) > maxLength:
            maxLength = len(seq)

    if maxLength > maxSize:
        maxLength = maxSize

    if isinstance(sequences[-1], list):
        x = np.ones((len(sequences), maxLength), dtype=dtype) * value
    else:
        shape = sequences[-1].shape
        size = [len(sequences)]

        size.extend(shape)

        size[1] = maxLength

        x = np.ones(size) * value

    for rowIdx, seq in enumerate(sequences):
        seqLen = len(seq) if len(seq) < maxLength else maxLength
        x[rowIdx, :seqLen] = seq

    return x

def softmaxPrediction(output):
    return output.data.max(1, keepdim=True)[1]

def loadActivationFunction(funcName):
    if funcName == 'relu':
        return F.relu
    elif funcName == 'tanh':
        return tanh
    elif funcName == 'sigmoid':
        return sigmoid
    else:
        raise ArgumentError(
            "Activation name %s is invalid. You should choose one of these: relu, tanh, sigmoid" % funcName)

def loadActivationClass(className):
    if className == 'relu':
        return nn.ReLU
    elif className == 'tanh':
        return nn.Tanh
    elif className == 'sigmoid':
        return nn.Sigmoid
    else:
        raise ArgumentError(
            "Activation name %s is invalid. You should choose one of these: relu, tanh, sigmoid" % className)
