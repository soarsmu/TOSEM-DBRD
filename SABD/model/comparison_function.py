"""
Comparison functions described in A COMPARE-AGGREGATE MODEL FOR MATCHING TEXT SEQUENCES Shuohang Wang 2017
"""

import torch
from torch import nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN, self).__init__()
        self.hidden = nn.Linear(input_size * 2, hidden_size)

    def forward(self, vectors, context):
        return F.relu(self.hidden(torch.cat([vectors, context], 2)))

class SubMultiNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SubMultiNN, self).__init__()
        self.hidden = nn.Linear(input_size * 2, hidden_size)

    def forward(self, vectors, context):
        subs = vectors - context
        squaredSub = subs * subs
        mult = vectors * context

        return F.relu(self.hidden(torch.cat([squaredSub, mult], 2)))


class Mult(nn.Module):

    def __init__(self):
        super(Mult, self).__init__()

    def forward(self, vectors, context):
        return vectors * context