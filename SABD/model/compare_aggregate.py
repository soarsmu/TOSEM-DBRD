import logging
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout, LayerNorm, BatchNorm1d, Sequential

from data.Embedding import Embedding
from model.attention import DotAttention, SelfAttention, GeneralAttention, ParikhAttention, Tan2016Attention
from model.basic_module import SortedRNNEncoder, CNN, meanVector, maxVector, \
    createEmbeddingLayer, ResidualLayer
from model.comparison_function import SubMultiNN, Mult, NN



class CADD(nn.Module):
    """
    Compare-aggregate model for duplicate bug report detection
    """

    def __init__(self, embedding_obj, categorical_encoder, opt, summary_opt, desc_opt, matching_opt, aggregate_opt):
        super(CADD, self).__init__()

        self.categorical_encoder = categorical_encoder
        self.desc_summarization = desc_opt['summarization']
        self.logger = logging.getLogger(CADD.__name__)

        # Extraction Stage
        self.sum_model_type = summary_opt['model_type']

        if summary_opt['model_type'] in ('lstm', 'gru'):
            self.sum_extractor = SortedRNNEncoder(summary_opt["model_type"], embedding_obj, summary_opt["hidden_size"], summary_opt["num_layers"], summary_opt["bidirectional"], summary_opt["update_embedding"], dropout=summary_opt["dropout"])
            sum_out_size = self.sum_extractor.getOutputSize()
            self.logger.info("CADD summary encoder: Sorted RNN")
        elif summary_opt['model_type'] == 'ELMo':
            # from allennlp.modules.elmo import Elmo
            #
            # options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            # weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            #
            # # Compute two different representation for each token.
            # # Each representation is a linear weighted combination for the
            # # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
            # self.sum_extractor = Elmo(options_file, weight_file, 1, dropout=summary_opt["dropout"],
            #                           requires_grad=summary_opt["fine_tune"], do_layer_norm=summary_opt["layer_norm"])
            # sum_out_size = self.sum_extractor.get_output_dim()
            sum_out_size = summary_opt["input_size"]
        elif summary_opt['model_type'] == 'BERT':
            from pytorch_transformers import BertModel
            self.sum_extractor = BertModel.from_pretrained("bert-base-uncased")

            for param in self.sum_extractor.parameters():
                param.requires_grad = summary_opt['fine_tune']

            sum_out_size = 768

            self.logger.info("CADD summary encoder: BERT")
        elif summary_opt['model_type'] == 'word_emd':
            self.sum_extractor, sum_out_size = createEmbeddingLayer(embedding_obj, summary_opt["update_embedding"])

            self.logger.info("CADD summary encoder: Embedding")
        elif summary_opt['model_type'] == 'residual':
            emb, sum_input_size = createEmbeddingLayer(embedding_obj, summary_opt["update_embedding"])
            self.sum_extractor = ResidualLayer(emb, F.tanh)

            sum_out_size = self.sum_extractor.getOutputSize()
            self.logger.info("CADD summary encoder: Residual layer")

        self.sum_dropout = Dropout(summary_opt["dropout"]) if summary_opt["dropout"] > 0.0 else None
        self.summary_norm = LayerNorm(sum_out_size) if summary_opt['layer_norm'] else None

        if not (summary_opt["update_embedding"] or desc_opt["update_embedding"]) and summary_opt['model_type'] not in [
            'ELMo', 'BERT']:
            if isinstance(self.sum_extractor, SortedRNNEncoder):
                # We shared the word embedding between the models, if they are not updating it
                descEmbedding = self.sum_extractor.rnnEncoder.embedding
            elif isinstance(self.sum_extractor, ResidualLayer):
                descEmbedding = self.sum_extractor.embedding
            elif isinstance(self.sum_extractor, torch.nn.modules.sparse.Embedding):
                descEmbedding = self.sum_extractor
            else:
                descEmbedding = None

            self.logger.info("CADD - Sharing embedding")
        else:
            descEmbedding = embedding_obj
            self.logger.info("CADD - Embedding are not shared")

        self.desc_model_type = summary_opt['model_type']

        if desc_opt['model_type'] in ('lstm', 'gru'):
            self.desc_extractor = SortedRNNEncoder(desc_opt["model_type"], descEmbedding, desc_opt["hidden_size"], desc_opt["num_layers"], desc_opt["bidirectional"], desc_opt["update_embedding"], dropout=desc_opt["dropout"])
            desc_out_size = self.desc_extractor.getOutputSize()
            self.logger.info("CADD description encoder: Sorted RNN")
        elif desc_opt['model_type'] == 'ELMo':
            # from allennlp.modules.elmo import Elmo
            #
            # options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            # weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            #
            # # Compute two different representation for each token.
            # # Each representation is a linear weighted combination for the
            # # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
            # self.desc_extractor = Elmo(options_file, weight_file, 1, dropout=desc_opt["dropout"],
            #                           requires_grad=desc_opt["fine_tune"], do_layer_norm=desc_opt["layer_norm"])
            # desc_out_size = self.desc_extractor.get_output_dim()

            desc_extractor = desc_opt["input_size"]
        elif desc_opt['model_type'] == 'BERT':
            # if not (desc_opt['fine_tune'] or summary_opt['fine_tune']) and isinstance(self.sum_extractor,
            #                                                                           BertModel):
            #     self.desc_extractor = self.sum_extractor
            # else:
            #     self.desc_extractor = BertModel.from_pretrained("bert-base-uncased")
            self.desc_extractor = BertModel.from_pretrained("bert-base-uncased")

            for param in self.desc_extractor.parameters():
                param.requires_grad = desc_opt['fine_tune']

            desc_out_size = 768

            self.logger.info("CADD summary encoder: BERT")
        elif desc_opt['model_type'] == 'word_emd':
            self.logger.info("CADD description encoder: Embedding Layer")
            if isinstance(descEmbedding, Embedding):
                self.desc_extractor, desc_out_size = createEmbeddingLayer(descEmbedding, desc_opt["update_embedding"])
            else:
                self.desc_extractor = self.sum_extractor
                desc_out_size = sum_out_size
        elif desc_opt['model_type'] == 'residual':
            if isinstance(descEmbedding, Embedding):
                descEmbedding, desc_input_size = createEmbeddingLayer(embedding_obj, desc_opt["update_embedding"])

            self.desc_extractor = ResidualLayer(descEmbedding, F.tanh)

            desc_out_size = self.desc_extractor.getOutputSize()
            self.logger.info("CADD description encoder: Residual layer")

        self.desc_dropout = Dropout(desc_opt["dropout"]) if desc_opt["dropout"] > 0.0 else None
        self.desc_norm = LayerNorm(desc_out_size) if desc_opt['layer_norm'] else None

        # Matching Stage
        self.categorical_hidden = nn.Linear(categorical_encoder.getOutputSize() * 2, matching_opt['categorical_hidden_layer'])
        self.categorical_dropout = Dropout(matching_opt['categorical_dropout']) if matching_opt['categorical_dropout'] > 0.0 else None

        self.categorical_norm = LayerNorm(matching_opt['categorical_hidden_layer']) if matching_opt[
            'categorical_layer_norm'] else None

        self.onlyCandidate = opt['only_candidate']
        anchor_cand_att = not self.onlyCandidate

        if matching_opt['attention'] is not None and matching_opt['attention'] != 'none':
            self.sum_sum_attention = self.create_attention_layer(matching_opt, sum_out_size, sum_out_size, anchor_cand_att)

            self.sum_sum_matching, sum_sum_size = self.create_matching_layer(matching_opt, sum_out_size)

            self.sum_sum_norm = LayerNorm(sum_sum_size) if matching_opt['layer_norm'] else None

            if matching_opt['cross_attention']:
                self.sum_desc_attention = self.create_attention_layer(matching_opt, sum_out_size, desc_out_size, True)
                self.sum_desc_matching, sum_desc_size = self.create_matching_layer(matching_opt, sum_out_size)

                self.sum_desc_norm = LayerNorm(sum_desc_size) if matching_opt['layer_norm'] else None
            else:
                self.sum_desc_attention = None
                self.sum_desc_matching = None
                sum_desc_size = 0

            self.desc_desc_attention = self.create_attention_layer(matching_opt, desc_out_size, desc_out_size, anchor_cand_att)
            self.desc_desc_matching, desc_desc_size = self.create_matching_layer(matching_opt, desc_out_size)

            self.desc_desc_norm = LayerNorm(desc_desc_size) if matching_opt['layer_norm'] else None

        else:
            self.sum_sum_matching = None
            self.desc_desc_matching = None
            desc_desc_size = desc_out_size
            sum_sum_size = sum_out_size
            sum_desc_size = 0

        self.agg_concat = aggregate_opt['concat']

        sum_matching_out = sum_out_size + sum_sum_size + sum_desc_size if aggregate_opt[
            'concat'] else sum_sum_size + sum_desc_size
        desc_matching_out = desc_out_size + desc_desc_size + sum_desc_size if aggregate_opt[
            'concat'] else desc_desc_size + sum_desc_size

        # Aggregate stage
        if aggregate_opt['model'] == 'cnn':
            self.sum_agg = CNN(sum_matching_out, aggregate_opt['window'], aggregate_opt['nfilters'])
            self.desc_agg = CNN(desc_matching_out, aggregate_opt['window'], aggregate_opt['nfilters'])
            self.min_sentences = max(aggregate_opt['window'])

            self.agg_sum_norm = LayerNorm(self.sum_agg.getOutputSize()) if aggregate_opt['layer_norm'] else None
            self.agg_desc_norm = LayerNorm(self.desc_agg.getOutputSize()) if aggregate_opt['layer_norm'] else None

            agg_out_size = self.sum_agg.getOutputSize() + self.desc_agg.getOutputSize()
        elif aggregate_opt['model'] == 'self_att':
            self.sum_agg = SelfAttention(sum_matching_out, aggregate_opt['hidden_size'])
            self.desc_agg = SelfAttention(desc_matching_out, aggregate_opt['hidden_size'])

            self.agg_sum_norm = LayerNorm(sum_matching_out) if aggregate_opt['layer_norm'] else None
            self.agg_desc_norm = LayerNorm(desc_matching_out) if aggregate_opt['layer_norm'] else None

            agg_out_size = sum_matching_out + desc_matching_out

            self.min_sentences = 0
        elif aggregate_opt['model'] == 'mean+max':
            agg_out_size = 2 * (sum_matching_out + desc_matching_out)
            self.min_sentences = 0
        elif aggregate_opt['model'] in ['lstm', 'gru']:
            self.sum_agg = SortedRNNEncoder(aggregate_opt["model"], sum_matching_out, aggregate_opt["hidden_size"], aggregate_opt["num_layers"], aggregate_opt["bidirectional"], dropout=aggregate_opt["dropout"])
            self.desc_agg = SortedRNNEncoder(aggregate_opt["model"], desc_matching_out, aggregate_opt["hidden_size"], aggregate_opt["num_layers"], aggregate_opt["bidirectional"], dropout=aggregate_opt["dropout"])
            self.pooling = aggregate_opt["pooling"]

            self.agg_sum_norm = LayerNorm(self.sum_agg.getOutputSize()) if aggregate_opt['layer_norm'] else None
            self.agg_desc_norm = LayerNorm(self.desc_agg.getOutputSize()) if aggregate_opt['layer_norm'] else None

            if aggregate_opt["pooling"] == 'self_att':
                self.sum_pooling = SelfAttention(self.sum_agg.getOutputSize(), aggregate_opt['self_att_hidden'])
                self.desc_pooling = SelfAttention(self.desc_agg.getOutputSize(), aggregate_opt['self_att_hidden'])

                agg_out_size = self.sum_agg.getOutputSize() + self.desc_agg.getOutputSize()
            elif aggregate_opt["pooling"] == 'mean+max':
                agg_out_size = 2 * (self.sum_agg.getOutputSize() + self.desc_agg.getOutputSize())
            else:
                raise Exception('Unknown parameter name %s' % aggregate_opt['pooling'])

            self.min_sentences = 0
        else:
            raise Exception('Unknown parameter name %s' % aggregate_opt['model'])

        self.aggregate_model = aggregate_opt['model']

        # Score stage
        linear_in_size = agg_out_size if self.onlyCandidate else 2 * agg_out_size
        self.matching_dropout = Dropout(matching_opt['dropout']) if matching_opt['dropout'] > 0.0 else None
        self.aggregation_dropout = Dropout(aggregate_opt['dropout']) if aggregate_opt['dropout'] > 0.0 else None
        self.classifier_dropout = Dropout(opt['dropout']) if opt['dropout'] > 0.0 else None

        self.h1 = nn.Linear(matching_opt['categorical_hidden_layer'] + linear_in_size, opt['hidden_size'])
        self.linear_output = nn.Linear(opt['hidden_size'], 2)

        if opt['layer_norm']:
            self.classifier_norm = LayerNorm(opt['hidden_size'])
        elif opt['batch_normalization']:
            self.classifier_norm = BatchNorm1d(opt['hidden_size'])
        else:
            self.classifier_norm = None

    def create_attention_layer(self, matching_opt, input_size, output_size, bidirectional):
        return GeneralAttention(input_size, output_size, bidirectional) if matching_opt["attention"] == 'general' else DotAttention(
            True, scale=1 / math.sqrt(input_size))

    def create_matching_layer(self, matching_opt, input_size):
        if matching_opt['comparison_function'] == 'sub_mult_nn':
            return SubMultiNN(input_size, matching_opt['comparison_hidden_size']), matching_opt[
                'comparison_hidden_size']
        elif matching_opt['name'] == 'mult':
            return torch.mul, input_size

    def encode(self, categorical_input, summary_input, description_input):
        """
        an extraction layer extracts the higher level features from the query and candidate reports.
        :param inputs:
        :return:
        """
        # Generate the representation of categorical data
        categorical_emb = self.categorical_encoder(*categorical_input)

        # Generate the summary embedding of all words
        if self.sum_model_type == 'lstm' or self.sum_model_type == 'gru':
            sumary_emb, _ = self.sum_extractor(*summary_input)

            # The number of words in each summary in the batch
            summary_len = summary_input[2]
        elif self.sum_model_type == 'ELMo':
            sumary_emb, summary_len = summary_input
        elif self.sum_model_type == 'BERT':
            sumary_emb = self.sum_extractor(summary_input[0], attention_mask=summary_input[1])[0]
            summary_len = summary_input[2]
        else:
            sumary_emb = self.sum_extractor(summary_input[0])

            # The number of words in each summary in the batch
            summary_len = summary_input[2]

        if self.sum_dropout is not None:
            sumary_emb = self.sum_dropout(sumary_emb)

        if self.summary_norm is not None:
            sumary_emb = self.summary_norm(sumary_emb)

        if self.desc_summarization:
            # Discontinued
            raise Exception("desc_summarization was discontinued")
            # descriptions, longestDesc, descLengths = description_input
            # outputs = []
            #
            # for ex in descriptions:
            #     descOut, _ = self.desc_rnn(*ex)
            #     descEmb = meanVector(descOut, ex[2])
            #
            #     if self.dropout:
            #         descEmb = self.dropout(descEmb)
            #
            #     # Padding with zero the outputs
            #     descLength = descEmb.size(0)
            #     if descLength < longestDesc:
            #         descEmb = F.pad(descEmb, (0, 0, 0, longestDesc - descLength), "constant", 0)
            #
            #     outputs.append(descEmb)
            #
            # description_emb = torch.stack(outputs, dim=0)
            #
            # if description_emb.size(1) < self.min_sentences:
            #     description_emb = F.pad(description_emb, (0, 0, 0, self.min_sentences - description_emb.size(1)), "constant", 0)
            #
            # description_len = descLengths
        else:
            # Generate the description embedding of all words
            # Generate the summary embedding of all words
            if self.desc_model_type == 'lstm' or self.desc_model_type == 'gru':
                description_emb, _ = self.desc_extractor(*description_input)
                # The number of words in each summary in the batch
                description_len = description_input[2]
            elif self.desc_model_type == 'ELMo':
                description_emb, description_len = description_input
            elif self.desc_model_type == 'BERT':
                description_emb = self.desc_extractor(description_input[0], attention_mask=description_input[1])[0]
                description_len = description_input[2]
            else:
                description_emb = self.desc_extractor(description_input[0])
                # The number of words in each summary in the batch
                description_len = description_input[2]

        if self.desc_dropout is not None:
            description_emb = self.desc_dropout(description_emb)

        if self.desc_norm is not None:
            description_emb = self.desc_norm(description_emb)

        return (sumary_emb, summary_len), (description_emb, description_len), categorical_emb

    def forward(self, anchorInputs, candidateInputs):
        # Encode input
        anchor_sum, anchor_desc, anchor_categorical = self.encode(*anchorInputs)
        candidate_sum, candidate_desc, candidate_categorical = self.encode(*candidateInputs)
        return self.similarity(
            anchor_sum, anchor_desc, anchor_categorical,
            candidate_sum, candidate_desc, candidate_categorical)

    def matching(self, attention, matching_layer, anchor, candidate, layer_norm):
        # Calcule the attention between the two outputs
        if attention.bidirectional:
            # Match the candidate and query
            anchor_context, candidate_context, anchor_att, candidate_att = attention(anchor, candidate)

            # Comparison between output and its context. Use the comparison functions proposed by Wang, 2016
            candidate_comparison = matching_layer(candidate, candidate_context)
            anchor_comparison = matching_layer(anchor, anchor_context)

            if layer_norm is not None:
                anchor_comparison = layer_norm(anchor_comparison)
                candidate_comparison = layer_norm(candidate_comparison)

            if self.matching_dropout:
                anchor_comparison = self.matching_dropout(anchor_comparison)
                candidate_comparison = self.matching_dropout(candidate_comparison)
        else:
            # Only match the candidate and its context vectors.
            candidate_context, candidate_att = attention(anchor, candidate)
            candidate_comparison = matching_layer(candidate, candidate_context)

            if layer_norm is not None:
                candidate_comparison = layer_norm(candidate_comparison)

            if self.matching_dropout:
                candidate_comparison = self.matching_dropout(candidate_comparison)

            anchor_comparison = None

        return anchor_comparison, candidate_comparison

    def similarity(self, anchor_sum, anchor_desc, anchor_categorical, candidate_sum, candidate_desc, candidate_categorical):
        anchor_sum_emb, anchor_sum_lengths = anchor_sum
        anchor_desc_emb, anchor_desc_lengths = anchor_desc

        cand_sum_emb, cand_sum_lengths = candidate_sum
        cand_desc_emb, cand_desc_lengths = candidate_desc

        attFtrsSumAnchor = [anchor_sum_emb]
        attFtrsDescAnchor = [anchor_desc_emb]

        attFtrsSumCand = [cand_sum_emb]
        attFtrsDescCand = [cand_desc_emb]

        # Comparison stage
        if self.sum_sum_matching:
            anchor_sum_sum, cand_sum_sum = self.matching(self.sum_sum_attention, self.sum_sum_matching, anchor_sum_emb,
            cand_sum_emb, self.sum_sum_norm)

            anchor_desc_desc, cand_desc_desc = self.matching(self.desc_desc_attention, self.desc_desc_matching,
            anchor_desc_emb, cand_desc_emb, self.desc_desc_norm)

            if self.sum_desc_attention:
                # todo: instead of comparing each of the fields using a different attention layers, we should add an information related to the field in the embeddings
                anchor_sum_desc, cand_desc_sum = self.matching(
                    self.sum_desc_attention, self.sum_desc_matching,
                    anchor_sum_emb, cand_desc_emb, self.sum_desc_norm)
                cand_sum_desc, anchor_desc_sum = self.matching(
                    self.sum_desc_attention, self.sum_desc_matching,
                    cand_sum_emb, anchor_desc_emb, self.sum_desc_norm)

                if anchor_sum_sum is not None:
                    anchor_sum_comparison = torch.cat([anchor_sum_sum, anchor_sum_desc], dim=2)
                    anchor_desc_comparison = torch.cat([anchor_desc_desc, anchor_desc_sum], dim=2)
                else:
                    anchor_sum_comparison = None
                    anchor_desc_comparison = None

                cand_sum_comparison = torch.cat([cand_sum_sum, cand_sum_desc], dim=2)
                cand_desc_comparison = torch.cat([cand_desc_desc, cand_desc_sum], dim=2)
            else:
                anchor_sum_comparison = anchor_sum_sum
                anchor_desc_comparison = anchor_desc_desc

                cand_sum_comparison = cand_sum_sum
                cand_desc_comparison = cand_desc_desc

        else:
            anchor_sum_comparison = anchor_sum_emb
            anchor_desc_comparison = anchor_desc_emb

            cand_sum_comparison = cand_sum_emb
            cand_desc_comparison = cand_desc_emb

        categorical_comparison = F.relu(self.categorical_hidden(torch.cat([anchor_categorical * candidate_categorical, (anchor_categorical - candidate_categorical) ** 2], 1)))

        if self.categorical_norm is not None:
            categorical_comparison = self.categorical_norm(categorical_comparison)

        if self.categorical_dropout is not None:
            categorical_comparison = self.categorical_dropout(categorical_comparison)

        # Aggregate stage
        agg_outputs = [categorical_comparison]

        agg_model = self.aggregate_model

        if self.agg_concat:
            anchor_sum_matching_emb = torch.cat((anchor_sum_emb, anchor_sum_comparison), 2)
            anchor_desc_matching_emb = torch.cat((anchor_desc_emb, anchor_desc_comparison), 2)

            cand_sum_matching_emb = torch.cat((cand_sum_emb, cand_sum_comparison), 2)
            cand_desc_matching_emb = torch.cat((cand_desc_emb, cand_desc_comparison), 2)
        else:
            anchor_sum_matching_emb = anchor_sum_comparison
            anchor_desc_matching_emb = anchor_desc_comparison

            cand_sum_matching_emb = cand_sum_comparison
            cand_desc_matching_emb = cand_desc_comparison

        if agg_model == 'mean+max':
            if anchor_sum_matching_emb is not None:
                # We use the comparison of the query and candidate
                # Aggregation Anchor summary
                sum_agg_emb = torch.cat([meanVector(anchor_sum_matching_emb, anchor_sum_lengths), maxVector(anchor_sum_matching_emb, anchor_sum_lengths)], 1)
                desc_agg_emb = torch.cat([meanVector(anchor_desc_matching_emb, anchor_desc_lengths), maxVector(anchor_desc_matching_emb, anchor_desc_lengths)], 1)
                agg_outputs.append(sum_agg_emb)
                agg_outputs.append(desc_agg_emb)

            sum_agg_emb = torch.cat([meanVector(cand_sum_matching_emb, cand_sum_lengths), maxVector(cand_sum_matching_emb, cand_sum_lengths)], 1)
            desc_agg_emb = torch.cat([meanVector(cand_desc_matching_emb, cand_desc_lengths), maxVector(cand_desc_matching_emb, cand_desc_lengths)], 1)

            agg_outputs.append(sum_agg_emb)
            agg_outputs.append(desc_agg_emb)
        elif agg_model == 'self_att':
            if anchor_sum_matching_emb is not None:
                # We use the comparison of the query and candidate
                # Aggregation anchor summary

                sum_agg_emb = self.sum_agg(anchor_sum_matching_emb)[0]
                desc_agg_emb = self.desc_agg(anchor_desc_matching_emb)[0]

                if self.agg_sum_norm:
                    sum_agg_emb = self.agg_sum_norm(sum_agg_emb)
                    desc_agg_emb = self.agg_desc_norm(desc_agg_emb)

                agg_outputs.append(sum_agg_emb)
                agg_outputs.append(desc_agg_emb)

            sum_agg_emb = self.sum_agg(cand_sum_matching_emb)[0]
            desc_agg_emb = self.desc_agg(cand_desc_matching_emb)[0]

            if self.agg_sum_norm:
                sum_agg_emb = self.agg_sum_norm(sum_agg_emb)
                desc_agg_emb = self.agg_desc_norm(desc_agg_emb)

            agg_outputs.append(sum_agg_emb)
            agg_outputs.append(desc_agg_emb)

        elif agg_model == 'lstm' or agg_model == 'gru':
            if anchor_sum_matching_emb is not None:
                # We use the comparison of the query and candidate
                # Aggregation anchor summary
                sum_agg_rnn_output, _ = self.sum_agg(anchor_sum_matching_emb, None, anchor_sum_lengths, len(anchor_sum_lengths))
                desc_agg_rnn_output, _ = self.desc_agg(anchor_desc_matching_emb, None, anchor_desc_lengths, len(anchor_desc_lengths))

                if self.agg_sum_norm:
                    sum_agg_rnn_output = self.agg_sum_norm(sum_agg_rnn_output)
                    desc_agg_rnn_output = self.agg_desc_norm(desc_agg_rnn_output)

                if self.pooling == 'self_att':
                    agg_outputs.append(self.sum_pooling(sum_agg_rnn_output)[0])
                    agg_outputs.append(self.desc_pooling(desc_agg_rnn_output)[0])
                elif self.pooling == 'mean+max':
                    agg_outputs.append(torch.cat([meanVector(sum_agg_rnn_output, anchor_sum_lengths),
                    maxVector(sum_agg_rnn_output, anchor_sum_lengths)], 1))
                    agg_outputs.append(torch.cat([meanVector(desc_agg_rnn_output, anchor_desc_lengths),
                    maxVector(desc_agg_rnn_output, anchor_desc_lengths)], 1))

            sum_agg_rnn_output, _ = self.sum_agg(cand_sum_matching_emb, None, cand_sum_lengths, len(cand_sum_lengths))
            desc_agg_rnn_output, _ = self.desc_agg(cand_desc_matching_emb, None, cand_desc_lengths, len(cand_desc_lengths))

            if self.agg_sum_norm:
                sum_agg_rnn_output = self.agg_sum_norm(sum_agg_rnn_output)
                desc_agg_rnn_output = self.agg_desc_norm(desc_agg_rnn_output)

            if self.pooling == 'self_att':
                agg_outputs.append(self.sum_pooling(sum_agg_rnn_output)[0])
                agg_outputs.append(self.desc_pooling(desc_agg_rnn_output)[0])
            elif self.pooling == 'mean+max':
                agg_outputs.append(torch.cat([meanVector(sum_agg_rnn_output, cand_sum_lengths), maxVector(sum_agg_rnn_output, cand_sum_lengths)], 1))
                agg_outputs.append(torch.cat([meanVector(desc_agg_rnn_output, cand_desc_lengths), maxVector(desc_agg_rnn_output, cand_desc_lengths)], 1))
        else:
            if anchor_sum_matching_emb is not None:
                # We use the comparison of the query and candidate
                # Aggregation anchor summary
                sum_agg_emb = self.sum_agg(anchor_sum_matching_emb)
                desc_agg_emb = self.desc_agg(anchor_desc_matching_emb)

                if self.agg_sum_norm:
                    sum_agg_emb = self.agg_sum_norm(sum_agg_emb)
                    desc_agg_emb = self.agg_desc_norm(desc_agg_emb)

                agg_outputs.append(sum_agg_emb)
                agg_outputs.append(desc_agg_emb)

            sum_agg_emb = self.sum_agg(cand_sum_matching_emb)
            desc_agg_emb = self.desc_agg(cand_desc_matching_emb)

            if self.agg_sum_norm:
                sum_agg_emb = self.agg_sum_norm(sum_agg_emb)
                desc_agg_emb = self.agg_desc_norm(desc_agg_emb)

            agg_outputs.append(sum_agg_emb)
            agg_outputs.append(desc_agg_emb)

        if self.aggregation_dropout is not None:
            for i in range(1, len(agg_outputs)):
                agg_outputs[i] = self.aggregation_dropout(agg_outputs[i])

        # Score stage
        x = torch.cat(agg_outputs, 1)
        x = F.relu(self.h1(x))

        if self.classifier_norm:
            x = self.classifier_norm(x)

        if self.classifier_dropout:
            x = self.classifier_dropout(x)

        return F.log_softmax(self.linear_output(x), dim=1)

    def getOutputSize(self):
        return self.dense.getOutputSize()


class SABD(nn.Module):
    """
    Compare-aggregate model for duplicate bug report detection
    """

    def __init__(self, embedding_obj, categorical_encoder, extractor_opt, matching_opt, aggregate_opt, classifier_opt, freq=False):
        super(SABD, self).__init__()

        self.categorical_encoder = categorical_encoder
        self.logger = logging.getLogger(CADD.__name__)

        # Create and set word encoder
        self.word_embedding, self.word_size = createEmbeddingLayer(embedding_obj, False)
        self.paddingId = self.word_embedding.padding_idx
        self.emb_dropout = Dropout(extractor_opt["emb_dropout"]) if extractor_opt["emb_dropout"] > 0.0 else None

        self.freq = freq
        freq_pos = 1 if freq else 0

        self.field_word_combination = extractor_opt['field_word_combination']

        if self.field_word_combination == 'cat':
            txt_field_emb_size = extractor_opt['txt_field_emb_size']
            self.logger.info("==> Field Word combination: CAT; field_size: {}".format(txt_field_emb_size))
            self.txt_field_embedding = nn.Embedding(3, txt_field_emb_size, padding_idx=0)
        elif self.field_word_combination == 'add':
            self.logger.info("==> Field Word combination: ADD; field_size: {}".format(self.word_size + freq_pos))
            self.txt_field_embedding = nn.Embedding(3, self.word_size + freq_pos, padding_idx=0)
            txt_field_emb_size = 0
        elif self.field_word_combination is None or self.field_word_combination == 'none':
            self.txt_field_embedding = None
            txt_field_emb_size = 0

        self.field_word_size = txt_field_emb_size + self.word_size + freq_pos
        self.extractor_model = extractor_opt['model']

        self.logger.info("==> Textual Encoder: use categorical data={}".format(extractor_opt['use_categorical']))
        self.extractor_use_categorical = extractor_opt['use_categorical']
        extractor_input_size = self.field_word_size + categorical_encoder.getOutputSize() if extractor_opt[
            'use_categorical'] else self.field_word_size

        self.logger.info("==> Textual Encoder: model={}, input_size={}, output_size={}, use categorical data={}".format(
            extractor_opt['model'], extractor_input_size, self.field_word_size, extractor_opt['use_categorical']))

        if extractor_opt['model'] in ('lstm', 'gru'):
            self.encoder = SortedRNNEncoder(extractor_opt["model"], self.field_word_size, extractor_opt["hidden_size"], 1, extractor_opt["bidirectional"])
            self.field_word_size = self.encoder.getOutputSize()
            self.logger.info("==> Encoder: Sorted RNN {} bi={}".format(extractor_opt["model"], extractor_opt["bidirectional"]))
        if extractor_opt['model'] in ('linear+lstm', 'linear+gru'):
            rnn_type =  extractor_opt['model'][-4:]

            self.linear_encoder = nn.Linear(extractor_input_size, self.field_word_size)
            self.encoder = SortedRNNEncoder(rnn_type, self.field_word_size, extractor_opt["hidden_size"], 1, extractor_opt["bidirectional"])
            self.field_word_size = self.encoder.getOutputSize()
            self.logger.info("==> Encoder: Residual FC + Sorted RNN {} bi={}".format(extractor_opt["model"], extractor_opt["bidirectional"]))
        elif extractor_opt['model'] == 'linear':
            self.encoder = nn.Linear(extractor_input_size, self.field_word_size)
            self.logger.info("==> Encoder: Residual FC")
        elif extractor_opt['model'] == 'highway':
            self.encoder = nn.Linear(extractor_input_size, self.field_word_size)
            self.gate = nn.Linear(extractor_input_size, self.field_word_size)

            self.logger.info("==> Encoder: highway")
        elif extractor_opt['model'] == 'word':
            pass

        self.extraction_dropout = Dropout(extractor_opt["dropout"]) if extractor_opt["dropout"] > 0.0 else None
        self.extraction_norm = LayerNorm(extractor_opt) if extractor_opt['layer_norm'] else None

        # Textual Matching Stage
        self.onlyCandidate = classifier_opt['only_candidate']
        anchor_cand_att = not self.onlyCandidate

        self.matching_type = matching_opt["type"]
        self.residual = matching_opt["residual"]
        self.logger.info("===> Compare: residual connection={}".format(self.residual))

        if matching_opt is None or matching_opt["type"] == 'none':
            self.soft_alignment_attention = None
            self.logger.info("==> Without Matching")
        elif matching_opt["type"] == 'mean':
            self.logger.info("==> Compare with the mean vector of the other report")
            self.soft_alignment_attention = None

            if matching_opt['comparison_func'] == 'submult+nn':
                self.logger.info("===> Compare: SubMultiNN")
                self.cmp_matching_attention = SubMultiNN(self.field_word_size, self.field_word_size)
            elif matching_opt['comparison_func'] == 'mult':
                self.logger.info("===> Compare: MULT")
                self.cmp_matching_attention = Mult()
            elif matching_opt['comparison_func'] == 'nn':
                self.logger.info("===> Compare: NN")
                self.cmp_matching_attention = NN(self.field_word_size, self.field_word_size)

            self.matching_norm = LayerNorm(self.field_word_size) if matching_opt['layer_norm'] else None
            self.matching_dropout = Dropout(matching_opt['dropout']) if matching_opt['dropout'] > 0.0 else None

            if not self.residual or not anchor_cand_att:
                raise Exception("matching_type=mean only accepted residual({}) and query+candidate({})".format(self.residual, anchor_cand_att))

        elif matching_opt["type"] == 'partial_compare':
            self.logger.info("==> Partial comparison (the compare-aggregate is kept)")
            attention_opt = matching_opt['attention']
            attention_hidden_size = self.field_word_size if attention_opt == 'dot_product' else matching_opt[
                'attention_hidden_size']
            scale = 1 / math.sqrt(attention_hidden_size) if matching_opt['scaled_attention'] else None

            if attention_opt == 'dot_product':
                self.logger.info(
                    "===> Attention: dot_product. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = DotAttention(False, scale)
            elif attention_opt == "tan2016":
                self.logger.info(
                    "===> Attention: tanh1016. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = Tan2016Attention(self.field_word_size, matching_opt['attention_hidden_size'])
            if matching_opt['comparison_func'] == 'submult+nn':
                self.logger.info("===> Compare: SubMultiNN")
                self.cmp_matching_attention = SubMultiNN(self.field_word_size, self.field_word_size)
            elif matching_opt['comparison_func'] == 'mult':
                self.logger.info("===> Compare: MULT")
                self.cmp_matching_attention = Mult()
            elif matching_opt['comparison_func'] == 'nn':
                self.logger.info("===> Compare: NN")
                self.cmp_matching_attention = NN(self.field_word_size, self.field_word_size)

            self.matching_norm = LayerNorm(self.field_word_size) if matching_opt['layer_norm'] else None
            self.matching_dropout = Dropout(matching_opt['dropout']) if matching_opt['dropout'] > 0.0 else None

            if not self.residual or not anchor_cand_att:
                raise Exception(
                    "matching_type=mean only accepted residual({}) and query+candidate({})".format(self.residual, anchor_cand_att))
        elif matching_opt["type"] == 'partial':
            attention_opt = matching_opt['attention']
            attention_hidden_size = self.field_word_size if attention_opt == 'dot_product' else matching_opt[
                'attention_hidden_size']
            scale = 1 / math.sqrt(attention_hidden_size) if matching_opt['scaled_attention'] else None

            self.logger.info("==> Partial Matching")

            if attention_opt == 'dot_product':
                self.logger.info(
                    "===> Attention: dot_product. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = DotAttention(False, scale)
            elif attention_opt == "tan2016":
                self.logger.info(
                    "===> Attention: tanh1016. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = Tan2016Attention(self.field_word_size, matching_opt['attention_hidden_size'])
        elif matching_opt["type"] == 'full':
            self.logger.info("==> Full Matching")
            attention_opt = matching_opt['attention']
            attention_hidden_size = self.field_word_size if attention_opt == 'dot_product' else matching_opt[
                'attention_hidden_size']
            scale = 1 / math.sqrt(attention_hidden_size) if matching_opt['scaled_attention'] else None

            if attention_opt == 'dot_product':
                self.logger.info(
                    "===> Attention: dot_product. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = DotAttention(anchor_cand_att, scale)
            elif attention_opt == "parikh":
                self.logger.info(
                    "===> Attention: parikh. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = ParikhAttention(self.field_word_size, matching_opt['attention_hidden_size'], nn.ReLU(), False,
                                                                anchor_cand_att, scale)
            elif attention_opt == "parikh_linear":
                self.logger.info(
                    "===> Attention: parikh_linear. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = ParikhAttention(self.field_word_size, matching_opt['attention_hidden_size'], None, False, anchor_cand_att, scale)
            elif attention_opt == "query_key":
                self.logger.info(
                    "===> Attention: query_key. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = ParikhAttention(self.field_word_size, matching_opt['attention_hidden_size'], None, True, anchor_cand_att, scale)
            elif attention_opt == "query_key_non_linear":
                self.logger.info(
                    "===> Attention: query_key_non_linear. scale: {}; hidden_size: {}".format(scale, attention_hidden_size))
                self.soft_alignment_attention = ParikhAttention(self.field_word_size, matching_opt['attention_hidden_size'], nn.ReLU(), True, anchor_cand_att, scale)
            if matching_opt['comparison_func'] == 'submult+nn':
                self.logger.info("===> Compare: SubMultiNN")
                self.cmp_matching_attention = SubMultiNN(self.field_word_size, self.field_word_size)
            elif matching_opt['comparison_func'] == 'mult':
                self.logger.info("===> Compare: MULT")
                self.cmp_matching_attention = Mult()
            elif matching_opt['comparison_func'] == 'nn':
                self.logger.info("===> Compare: NN")
                self.cmp_matching_attention = NN(self.field_word_size, self.field_word_size)

            self.matching_norm = LayerNorm(self.field_word_size) if matching_opt['layer_norm'] else None
            self.matching_dropout = Dropout(matching_opt['dropout']) if matching_opt['dropout'] > 0.0 else None

        # Aggregate stage
        if aggregate_opt['model'] == 'cnn':
            self.logger.info("==> Aggregate: CNN")
            self.agg = CNN(self.field_word_size, aggregate_opt['window'], aggregate_opt['nfilters'])
            self.min_sentences = max(aggregate_opt['window'])

            self.agg_out_size = self.agg.getOutputSize()
        elif aggregate_opt['model'] == 'self_att':
            self.logger.info("==> Aggregate: self_att(hidden_size={}, n_hops={})".format(aggregate_opt['hidden_size'], aggregate_opt['n_hops']))
            self.agg = SelfAttention(self.field_word_size, aggregate_opt['hidden_size'], aggregate_opt['n_hops'])

            self.agg_out_size = self.agg.getOutputSize()
            self.min_sentences = 0
        elif aggregate_opt['model'] in ('lstm', 'gru'):
            self.agg = SortedRNNEncoder(aggregate_opt["model"], self.field_word_size, aggregate_opt['hidden_size'], 1, aggregate_opt["bidirectional"])
            self.agg_out_size = self.agg.getOutputSize() * 2
            self.logger.info("==> Aggregate: Sorted RNN {} bi={}".format(aggregate_opt["model"], aggregate_opt["bidirectional"]))
        elif aggregate_opt['model'] == 'mean+max':
            self.logger.info("==> Aggregate: mean+max)")
            self.agg_out_size = 2 * self.field_word_size
            self.min_sentences = 0
        elif aggregate_opt['model'] == 'none':
            self.logger.info("==> Aggregate: None)")
            self.agg_out_size = self.field_word_size
        else:
            raise Exception('Unknown parameter name %s' % aggregate_opt['model'])

        self.aggregate_model = aggregate_opt['model']

        # Compare categorical embedding of each text
        if categorical_encoder is not None:
            self.hadamard_diff_categorical = classifier_opt["hadamard_diff_categorical"]
            categorical_input_size = categorical_encoder.getOutputSize() * 2 * (int(self.hadamard_diff_categorical) + 1)

            self.categorical_dropout = Dropout(classifier_opt['categorical_dropout']) if classifier_opt['categorical_dropout'] > 0.0 else None

            if classifier_opt['categorical_hidden_layer'] is None or classifier_opt['categorical_hidden_layer'] < 1:
                self.categorical_hidden = None
                categorical_cmp_size = categorical_input_size

                self.logger.info("==> Classification: categorical embedding hadamard_diff={}, dropout={}".format(
                    self.hadamard_diff_categorical, classifier_opt['categorical_dropout']))
            else:
                self.categorical_hidden = nn.Linear(categorical_input_size, classifier_opt['categorical_hidden_layer'])
                categorical_cmp_size = classifier_opt['categorical_hidden_layer']

                self.logger.info(
                    "==> Classification: Categorical FC({},{}), hadamard_diff={}, dropout={}".format(
                        self.categorical_hidden.in_features, self.categorical_hidden.out_features,
                        self.hadamard_diff_categorical, classifier_opt['categorical_dropout']))


        else:
            self.categorical_hidden = None
            categorical_cmp_size = 0

        # Compare report embeddings related to textual data
        self.hadamard_diff_textual = classifier_opt['hadamard_diff_textual']
        m = 2 if self.hadamard_diff_textual and not self.onlyCandidate else 0
        n = 1 if self.onlyCandidate else 2

        textual_input_size = n * self.agg_out_size + m * self.agg_out_size

        if classifier_opt['textual_hidden_layer'] is None or classifier_opt['textual_hidden_layer'] < 1 \
                or self.onlyCandidate:
            self.textual_hidden = None
            textual_cmp_size = textual_input_size

            self.logger.info("==> Classification: report embedding hadamard_diff={},dropout={}".format(
                self.hadamard_diff_textual, aggregate_opt['dropout']))
        else:
            self.textual_hidden = nn.Linear(textual_input_size, classifier_opt['textual_hidden_layer'])
            textual_cmp_size = classifier_opt['textual_hidden_layer']

            self.logger.info("==> Classification: Textual FC({},{}) hadamard_diff={},dropout={}".format(
                self.textual_hidden.in_features, self.textual_hidden.out_features, self.hadamard_diff_textual,
                aggregate_opt['dropout']))

        # Score stage
        self.aggregation_dropout = Dropout(aggregate_opt['dropout']) if aggregate_opt['dropout'] > 0.0 else None

        seq = []
        last = categorical_cmp_size + textual_cmp_size
        self.logger.info("==> Only candidate embedding: {}".format(self.onlyCandidate))

        for currentSize in classifier_opt['hidden_size']:
            seq.append(nn.Linear(last, currentSize))

            if classifier_opt['layer_norm']:
                seq.append(LayerNorm(currentSize))
            elif classifier_opt['batch_normalization']:
                seq.append(BatchNorm1d(currentSize))
            else:
                self.classifier_norm = None

            seq.append(nn.ReLU())

            if classifier_opt['dropout'] > 0.0:
                seq.append(nn.Dropout(classifier_opt['dropout']))

            self.logger.info("==> Create Hidden Layer (%d,%d) in the classifier" % (last, currentSize))
            last = currentSize

        seq.append(nn.Linear(last, 1))

        if classifier_opt['output_act'] == 'sigmoid':
            seq.append(nn.Sigmoid())
        elif classifier_opt['output_act'] == 'tanh':
            seq.append(nn.Tanh())

        self.classifier = Sequential(*seq)

    def encode(self, *inputs):
        """
        an extraction layer extracts the higher level features from the query and candidate reports.
        :param inputs:
        :return:
        """
        if self.categorical_encoder is not None:
            categorical_input, textual_input = inputs
        else:
            categorical_input = None
            textual_input = inputs[0]

        word_input = textual_input[0]
        field_info = textual_input[1]
        tf_input = textual_input[2]
        lengths = textual_input[3]

        mask = (word_input != self.paddingId).float()

        # Generate the representation of categorical data
        if categorical_input is not None:
            categorical_emb = self.categorical_encoder(*categorical_input)
        else:
            categorical_emb = None

        # Generate the summary embedding of all words
        word_emb = self.word_embedding(word_input)

        if self.field_word_combination == 'cat':
            field_emb = self.txt_field_embedding(field_info)
            if self.freq:
                field_word_emb = torch.cat([field_emb, word_emb, tf_input.unsqueeze(2)], dim=2)
            else:
                field_word_emb = torch.cat([field_emb, word_emb], dim=2)

        elif self.freq:
            field_word_emb = torch.cat([word_emb, tf_input.unsqueeze(2)], dim=2)
        else:
            field_word_emb = word_emb


        if self.emb_dropout is not None:
            field_word_emb = self.emb_dropout(field_word_emb)

        ext_model = self.extractor_model

        if self.extractor_use_categorical:
            expanded_categorical = categorical_emb.unsqueeze(1).expand((-1, field_word_emb.shape[1], -1))
            extractor_input = torch.cat([expanded_categorical, field_word_emb], dim=2)
        else:
            extractor_input = field_word_emb

        if ext_model == 'linear':
            encoder_output = F.relu(self.encoder(extractor_input)) + field_word_emb
            apply_mask = True
        elif ext_model == 'highway':
            gate = torch.sigmoid(self.gate(extractor_input))
            encoder_output = F.relu(self.encoder(extractor_input)) * gate + (1 - gate) * field_word_emb
            apply_mask = True
        elif ext_model == 'word':
            encoder_output = field_word_emb
            apply_mask = False
        elif ext_model in ('gru', 'lstm'):
            encoder_output = self.encoder(extractor_input, None, lengths)[0]
            apply_mask = False
        elif ext_model in ('linear+gru', 'linear+lstm'):
            fc_output = F.relu(self.linear_encoder(extractor_input)) + field_word_emb
            encoder_output = self.encoder(fc_output, None, lengths)[0]
            apply_mask = False

        if self.field_word_combination == 'add':
            encoder_output = encoder_output + self.txt_field_embedding(field_info)

        if apply_mask:
            final_output = encoder_output * mask.unsqueeze(2)
        else:
            final_output = encoder_output

        if self.extraction_norm is not None:
            final_output = self.extraction_norm(final_output)

        if self.extraction_dropout is not None:
            final_output = self.extraction_dropout(final_output)

        return final_output, lengths, categorical_emb, mask

    def forward(self, anchorInputs, candidateInputs, non_candidate_input=None):
        # Encode input
        anchor, anchor_len, anchor_categorical, anchor_mask = self.encode(*anchorInputs)
        candidate, cand_len, candidate_categorical, cand_mask = self.encode(*candidateInputs)

        sim_pair = self.similarity(anchor, anchor_len, anchor_categorical, anchor_mask, candidate, cand_len, candidate_categorical, cand_mask)

        if non_candidate_input is None:
            return sim_pair

        non_candidate, non_cand_len, non_candidate_categorical, non_cand_mask = self.encode(*non_candidate_input)

        sim_neg_pair = self.similarity(anchor, anchor_len, anchor_categorical, anchor_mask, non_candidate, non_cand_len, non_candidate_categorical, non_cand_mask)

        return sim_pair, sim_neg_pair

    def matching(self, attention, matching_layer, anchor, candidate, anchor_mask, cand_mask):
        # Calcule the attention between the two outputs
        if attention.bidirectional:
            # Match the candidate and query
            anchor_context, candidate_context, anchor_att, candidate_att = attention(anchor, candidate, anchor_mask, cand_mask)

            # Comparison between output and its context. Use the comparison functions proposed by Wang, 2016
            candidate_comparison = matching_layer(candidate, candidate_context)
            anchor_comparison = matching_layer(anchor, anchor_context)

            anchor_comparison = anchor_comparison * anchor_mask.unsqueeze(2)
        else:
            # Only match the candidate and its context vectors.
            candidate_context, candidate_att = attention(anchor, candidate, anchor_mask, cand_mask)
            candidate_comparison = matching_layer(candidate, candidate_context)

            anchor_comparison = None

        candidate_comparison = candidate_comparison * cand_mask.unsqueeze(2)

        return anchor_comparison, candidate_comparison

    def similarity(self, anchor, anchor_len, anchor_categorical, anchor_mask, candidate, cand_len, candidate_categorical, candidate_mask):
        # Comparison
        if self.matching_type is None or self.matching_type == 'none':
            anchor_after_cmp = anchor
            cand_after_cmp = candidate

            anchor_comparison = None if self.onlyCandidate else True
        elif self.matching_type == 'mean':
            # Generate a fixed representation of the candidate and anchor
            anchor_fixed_representation = meanVector(anchor, anchor_len, True)
            candidate_fixed_representation = meanVector(candidate, cand_len, True)

            # Comparison between output and its context. Use the comparison functions proposed by Wang, 2016
            candidate_comparison = self.cmp_matching_attention(candidate, anchor_fixed_representation)
            anchor_comparison = self.cmp_matching_attention(anchor, candidate_fixed_representation)

            anchor_comparison = anchor_comparison * anchor_mask.unsqueeze(2)
            candidate_comparison = candidate_comparison * candidate_mask.unsqueeze(2)

            cand_after_cmp  = candidate + candidate_comparison
            anchor_after_cmp = anchor + anchor_comparison

            if self.matching_norm is not None:
                anchor_after_cmp = self.matching_norm(anchor_after_cmp)
                cand_after_cmp = self.matching_norm(cand_after_cmp)

            if self.matching_dropout is not None:
                anchor_after_cmp = self.matching_dropout(anchor_after_cmp)
                cand_after_cmp = self.matching_dropout(cand_after_cmp)

        elif self.matching_type == 'partial_compare':
            # Generate a fixed representation of the candidate and anchor
            anchor_fixed_representation = meanVector(anchor, anchor_len, True)
            candidate_fixed_representation = meanVector(candidate, cand_len, True)

            # Compare the sequence others with the fixed representation. The final embedding of the report will be weighted average of these words.
            cand_context, _ = self.soft_alignment_attention(anchor, candidate_fixed_representation, anchor_mask)
            anchor_context, _ = self.soft_alignment_attention(candidate, anchor_fixed_representation, candidate_mask)

            # Comparison between output and its context.
            candidate_comparison = self.cmp_matching_attention(candidate, cand_context)
            anchor_comparison = self.cmp_matching_attention(anchor, anchor_context)

            anchor_comparison = anchor_comparison * anchor_mask.unsqueeze(2)
            candidate_comparison = candidate_comparison * candidate_mask.unsqueeze(2)

            cand_after_cmp = candidate + candidate_comparison
            anchor_after_cmp = anchor + anchor_comparison

            if self.matching_norm is not None:
                anchor_after_cmp = self.matching_norm(anchor_after_cmp)
                cand_after_cmp = self.matching_norm(cand_after_cmp)

            if self.matching_dropout is not None:
                anchor_after_cmp = self.matching_dropout(anchor_after_cmp)
                cand_after_cmp = self.matching_dropout(cand_after_cmp)

        elif self.matching_type == 'full':
            anchor_comparison, cand_comparison = self.matching(self.soft_alignment_attention, self.cmp_matching_attention, anchor, candidate, anchor_mask, candidate_mask)
            if self.residual:
                cand_after_cmp = candidate + cand_comparison
            else:
                cand_after_cmp = cand_comparison

            if self.matching_norm is not None:
                cand_after_cmp = self.matching_norm(cand_after_cmp)

            if self.matching_dropout is not None:
                cand_after_cmp = self.matching_dropout(cand_after_cmp)

            if anchor_comparison is not None:
                if self.residual:
                    anchor_after_cmp = anchor + anchor_comparison
                else:
                    anchor_after_cmp = anchor_comparison

                if self.matching_norm is not None:
                    anchor_after_cmp = self.matching_norm(anchor_after_cmp)

                if self.matching_dropout is not None:
                    anchor_after_cmp = self.matching_dropout(anchor_after_cmp)
        elif self.matching_type == 'partial':
            # Generate a fixed representation of the candidate and anchor
            anchor_fixed_representation = meanVector(anchor, anchor_len, True)
            candidate_fixed_representation = meanVector(candidate, cand_len, True)

            # Compare the sequence others with the fixed representation. The final embedding of the report will be weighted average of these words.
            anchor_weighted_avg, anchor_att = self.soft_alignment_attention(anchor, candidate_fixed_representation, anchor_mask)
            cand_weighted_avg, candidate_att = self.soft_alignment_attention(candidate, anchor_fixed_representation, candidate_mask)

            anchor_emb = anchor_weighted_avg.squeeze(1)
            cand_emb = cand_weighted_avg.squeeze(1)

        # Aggregate stage
        agg_outputs = []
        agg_model = self.aggregate_model

        if agg_model == 'mean+max':
            if anchor_comparison is not None:
                # We use the comparison of the query and candidate
                # Aggregation Anchor summary
                anchor_emb = torch.cat([meanVector(anchor_after_cmp, anchor_len), maxVector(anchor_after_cmp, anchor_len)], 1)

                agg_outputs.append(anchor_emb)

            cand_emb = torch.cat([meanVector(cand_after_cmp, cand_len), maxVector(cand_after_cmp, cand_len)], 1)

            agg_outputs.append(cand_emb)
        elif agg_model == 'self_att':
            if anchor_comparison is not None:
                # We use the comparison of the query and candidate
                # Aggregation anchor summary
                anchor_emb = self.agg(anchor_after_cmp, anchor_mask)[0].view(anchor_after_cmp.size(0), self.agg_out_size) 
                agg_outputs.append(anchor_emb)

            cand_emb = self.agg(cand_after_cmp, candidate_mask)[0].view(cand_after_cmp.size(0), self.agg_out_size)
            agg_outputs.append(cand_emb)
    
        elif agg_model in ('lstm','gru'):
            if anchor_comparison is not None:
                # We use the comparison of the query and candidate
                # Aggregation anchor summary
                rnn_output = self.agg(anchor_after_cmp,None, anchor_len)[0]
                anchor_emb = torch.cat([meanVector(rnn_output, anchor_len),
                                        maxVector(rnn_output, anchor_len)], 1)
                agg_outputs.append(anchor_emb)

            rnn_output = self.agg(cand_after_cmp, None, cand_len)[0]
            cand_emb = torch.cat([meanVector(rnn_output, cand_len),
                                    maxVector(rnn_output, cand_len)], 1)
            agg_outputs.append(cand_emb)
        elif agg_model == 'cnn':
            if anchor_comparison is not None:
                # We use the comparison of the query and candidate
                # Aggregation anchor summary
                anchor_emb = self.agg(anchor_after_cmp)
                agg_outputs.append(anchor_emb)

            cand_emb = self.agg(cand_after_cmp)
            agg_outputs.append(cand_emb)
        elif agg_model == 'none' or agg_model is None:
            agg_outputs.append(anchor_emb)
            agg_outputs.append(cand_emb)

        # Dropout textual embedding
        if self.aggregation_dropout is not None:
            for i in range(len(agg_outputs)):
                agg_outputs[i] = self.aggregation_dropout(agg_outputs[i])

        # Comparison of the categorical embeddings
        if self.categorical_encoder is None:
            categorical_input = None
        else:
            categorical_input = [anchor_categorical, candidate_categorical]

            if self.categorical_dropout is not None:
                for i in range(len(categorical_input)):
                    categorical_input[i] = self.categorical_dropout(categorical_input[i])

            if self.hadamard_diff_categorical:
                categorical_input.append(anchor_categorical * candidate_categorical)
                categorical_input.append((anchor_categorical - candidate_categorical) ** 2)

        if self.categorical_hidden is not None:
            categorical_comparison = [F.relu(self.categorical_hidden(torch.cat(categorical_input, 1)))]
        else:
            categorical_comparison = categorical_input

        # Comparison of the textual embeddings
        if self.hadamard_diff_textual and anchor_comparison is not None:
            agg_outputs.append((anchor_emb - cand_emb) ** 2)
            agg_outputs.append(anchor_emb * cand_emb)

        if self.textual_hidden is None:
            textual_comparison = agg_outputs
        else:
            textual_comparison = [F.relu(self.textual_hidden(torch.cat(agg_outputs, 1)))]

        # Classification
        classification_inputs = textual_comparison
        if categorical_comparison is not None:
            classification_inputs.extend(categorical_comparison)

        x = torch.cat(classification_inputs, 1)

        return self.classifier(x)

    def getOutputSize(self):
        return self.dense.getOutputSize()