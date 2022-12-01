import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, bidirectional=False):
        """

        :param bidirectional: if it is true, so you calculate the attention using the encoder outputs for each decoder input
        and the vice-versa
        """
        super(Attention, self).__init__()
        self.bidirectional = bidirectional

    def forward(self, similarity_matrix, encoder_seq, decoder_seq, encoder_mask, decoder_mask):
        """

        :param similarity_matrix: decoder_seq_size X encoder_seq_size
        :param encoder_seq:
        :param decoder_seq:
        :return:
        """
        batch_size, encoder_len, _ = encoder_seq.size()
        decoder_len = decoder_seq.size(1)

        if encoder_mask is not None:
            similarity_matrix = similarity_matrix + (1.0 - encoder_mask.unsqueeze(1)) * -10000.00

        # Normalize the similarity score to be between 0 and 1
        decoder_att = F.softmax(similarity_matrix.view(-1, encoder_len), dim=1).view(batch_size, decoder_len, encoder_len)

        # Context_t = \sum_{i=1} attention[t,i] * encoder_output[i] where t is decoder output position
        # (batch,decoder_seq_len, encoder_seq_len) * (batch, encoder_seq_len, hidden_size) = (batch,decoder_seq_len, hidden_size)
        decoder_ctx = torch.bmm(decoder_att, encoder_seq)

        if self.bidirectional:
            transposed_similarity = similarity_matrix.transpose(1, 2).contiguous()

            if decoder_mask is not None:
                transposed_similarity = transposed_similarity + (1.0 - decoder_mask.unsqueeze(1)) * -10000.00

            # Encoder attention
            encoder_att = F.softmax(transposed_similarity.view(-1, decoder_len), dim=1).view(batch_size, encoder_len, decoder_len)

            # Context_t = \sum_{i=1} attention[t,i] * encoder_output[i] where t is decoder output position
            # (batch,decoder_seq_len, encoder_seq_len) * (batch, encoder_seq_len, hidden_size) = (batch,decoder_seq_len, hidden_size)
            encoder_ctx = torch.bmm(encoder_att, decoder_seq)

            return encoder_ctx, decoder_ctx, encoder_att, decoder_att

        return decoder_ctx, decoder_att


class GeneralAttention(Attention):
    """
    Luong2015

    alpha = decoder_last_output · W · encoder_ouput
    """

    def __init__(self, input_size, output_size, bidirectional=False):
        """

        :param input_size: input size that the hidden layer expects
        :param output_size: output size of the hidden layer

        """
        super(GeneralAttention, self).__init__(bidirectional)
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, encoder_seq, decoder_seq, encoder_mask=None, decoder_mask=None):
        # pass the encoder output to a linear layer  (batch_size,encoder_seq_len, hidden_size)
        x = self.linear(encoder_seq)

        # transpose decoder_seq (batch_size, encoder_seq_len, hidden_size) => (batch_size, hidden_size, encoder_seq_len)
        x = x.transpose(1, 2)

        # Peform dot product between x and decoder output (batch,decoder_seq_len,hidden_size) * (batch,hidden_size, encoder_seq_len) = (batch, decoder_seq_len, encoder_out)
        simarity_matrix = torch.bmm(decoder_seq, x)

        return super().forward(simarity_matrix, encoder_seq, decoder_seq, encoder_mask, decoder_mask)


class SelfAttention(nn.Module):
    """
    A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING Zhouhan Lin 2017
    """

    def __init__(self, inputSize, hiddenSize, n_hops=1):
        super(SelfAttention, self).__init__()

        self.n_hops = n_hops
        self.inputSize = inputSize
        self.W = nn.Linear(inputSize, hiddenSize, bias=False)
        self.wv = nn.Linear(hiddenSize, n_hops, bias=False)
        # self.wv.data.data.uniform(-0.1, 0.1)

    def forward(self, inputs, mask=None):
        x = torch.tanh(self.W(inputs))
        similarityMatrix = self.wv(x).transpose(1, 2).contiguous()

        if mask is not None:
            similarityMatrix = similarityMatrix + (1.0 - mask.unsqueeze(1)) * -10000

        # Calculate the attention coeff.
        batchSize, seqLen, vecSize = inputs.size()
        attention = F.softmax(similarityMatrix.view(-1, seqLen), dim=1).view(batchSize, self.n_hops, seqLen)

        return torch.bmm(attention, inputs), attention

    def getOutputSize(self):
        return self.n_hops * self.inputSize


class Tan2016Attention(nn.Module):
    """
    This attention is based on LSTM-BASED DEEP LEARNING MODELS FOR NON- FACTOID ANSWER SELECTION tanh 2016
    """

    def __init__(self, input_size, hidden_size):
        super(Tan2016Attention, self).__init__()

        self.f_decoder = nn.Linear(input_size, hidden_size, bias=False)
        self.f_encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, sequence, fixed_representation, mask=None):
        x = torch.tanh(self.f_decoder(fixed_representation) + self.f_encoder(sequence))
        similarity_matrix = self.wv(x).transpose(1, 2).contiguous()

        if mask is not None:
            similarity_matrix = similarity_matrix + (1.0 - mask.unsqueeze(1)) * -10000

        # Calculate the attention coeff.
        batchSize, seqLen, vecSize = sequence.size()
        attention = F.softmax(similarity_matrix.view(-1, seqLen), dim=1).view(batchSize, 1, seqLen)

        return torch.bmm(attention, sequence), attention


class ParikhAttention(Attention):
    """
    This attention is based on A Decomposable Attention Model for Natural Language Inference Ankur P. Parikh.
    There is an option that allow to apply different function  for encoder and decoder inputs before calculate the similarity.
    """

    def __init__(self, input_size, hidden_size, activation=nn.ReLU(), distinct_func=False, bidirectional=False, scale=None):
        super(ParikhAttention, self).__init__(bidirectional)
        self.scale = scale

        # Decoder is the candidate and encoder is the query
        if activation is None:
            self.f_decoder = nn.Linear(input_size, hidden_size, bias=False)
        else:
            self.f_decoder = nn.Sequential(nn.Linear(input_size, hidden_size), activation)

        if distinct_func:
            if activation is None:
                self.f_encoder = nn.Linear(input_size, hidden_size, bias=False)
            else:
                self.f_encoder = nn.Sequential(nn.Linear(input_size, hidden_size), activation)
        else:
            self.f_encoder = self.f_decoder

    def forward(self, encoder_seq, decoder_seq, encoder_mask=None, decoder_mask=None):
        # Transform decoder input
        transformed_decoder_seq = self.f_decoder(decoder_seq)

        # Transform encoder input
        transformed_encoder_seq = self.f_encoder(encoder_seq)

        # transpose encoder input (batch_size, secOut_seq_len, dim) => (batch_size, dim, secOut_seq_len)
        encoder_transposed = transformed_encoder_seq.transpose(1, 2)

        # Perform dot product between encoder_seq and decoder_seq (batch,encoder_seq_len,dim) * (batch,dim, decoder_seq_len) = (batch, encoder_seq_len, decoder_seq_len)
        similarity_matrix = torch.bmm(transformed_decoder_seq, encoder_transposed)

        if self.scale:
            similarity_matrix = similarity_matrix * self.scale

        return super().forward(similarity_matrix, encoder_seq, decoder_seq, encoder_mask, decoder_mask)


class DotAttention(Attention):
    """
    Luong2015

    alpha = decoder_last_output · encoder_ouput
    """

    def __init__(self, bidirectional=False, scale=None):
        super(DotAttention, self).__init__(bidirectional)

        self.scale = scale

    def forward(self, encoder_seq, decoder_seq, encoder_mask=None, decoder_mask=None):
        # transpose output2 (batch_size, secOut_seq_len, dim) => (batch_size, dim, secOut_seq_len)
        encoder_transposed = encoder_seq.transpose(1, 2)

        # Peform dot product between encoder_seq and decoder_seq (batch,encoder_seq_len,dim) * (batch,dim, decoder_seq_len) = (batch, encoder_seq_len, decoder_seq_len)
        similarity_matrix = torch.bmm(decoder_seq, encoder_transposed)

        if self.scale:
            similarity_matrix = similarity_matrix * self.scale

        return super().forward(similarity_matrix, encoder_seq, decoder_seq, encoder_mask, decoder_mask)
