import codecs
import numpy as np
from argparse import ArgumentError

from nltk import WhitespaceTokenizer

from data.Embedding import Embedding
from data.Lexicon import Lexicon
from data.input_handler import TextCNNInputHandler, RNNInputHandler, BasicInputHandler
from data.preprocessing import loadFilters, MultiLineTokenizer, PreprocessingCache, SummaryDescriptionPreprocessor, \
    DescriptionPreprocessor, SummaryPreprocessor
from model.attention import SelfAttention
from model.basic_module import TextCNN, MultilayerDense, SortedRNNEncoder, RNNFixedOuput, RNN_Self_Attention, \
    Dense_Self_Attention
from model.siamese import CategoricalEncoder, WordMean
from util.data_util import createCategoricalPreprocessorAndLexicons
from util.torch_util import loadActivationFunction, loadActivationClass


def load_embedding(opts, paddingSym):
    if opts["lexicon"]:
        emb = np.load(opts["word_embedding"])

        lexicon = Lexicon(unknownSymbol=None)
        with codecs.open(opts["lexicon"]) as f:
            for l in f:
                lexicon.put(l.strip())

        lexicon.setUnknown("UUUKNNN")
        paddingId = lexicon.getLexiconIndex(paddingSym)
        embedding = Embedding(lexicon, emb, paddingIdx=paddingId)
    elif opts["word_embedding"]:
        # todo: Allow use embeddings and other representation
        lexicon, embedding = Embedding.fromFile(opts['word_embedding'], 'UUUKNNN', hasHeader=False, paddingSym=paddingSym)

    return lexicon, embedding

def processSumDescParam(sum_desc_opts, bugReportDatabase, inputHandlers, preprocessors, encoders, cacheFolder, databasePath, logger, paddingSym):
    # Use summary and description (concatenated) to address this problem
    logger.info("Using summary and description information.")
    # Loading word embedding
    lexicon, embedding =  load_embedding(sum_desc_opts, paddingSym)
    logger.info("Lexicon size: %d" % (lexicon.getLen()))
    logger.info("Word Embedding size: %d" % (embedding.getEmbeddingSize()))
    paddingId = lexicon.getLexiconIndex(paddingSym)
    # Loading Filters
    filters = loadFilters(sum_desc_opts['filters'])
    # Tokenizer
    if sum_desc_opts['tokenizer'] == 'default':
        logger.info("Use default tokenizer to tokenize summary+description information")
        tokenizer = MultiLineTokenizer()
    elif sum_desc_opts['tokenizer'] == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary+description information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            sum_desc_opts['tokenizer'])
    arguments = (
        databasePath, sum_desc_opts['word_embedding'],
        ' '.join(sorted([fil.__class__.__name__ for fil in filters])),
        sum_desc_opts['tokenizer'], "summary_description")
    cacheSumDesc = PreprocessingCache(cacheFolder, arguments)
    sumDescPreprocessor = SummaryDescriptionPreprocessor(lexicon, bugReportDatabase, filters, tokenizer, paddingId, cacheSumDesc)
    preprocessors.append(sumDescPreprocessor)
    if sum_desc_opts['encoder_type'] == 'cnn':
        windowSizes = sum_desc_opts.get('window_sizes', [3])
        nFilters = sum_desc_opts.get('nfilters', 100)
        updateEmb = sum_desc_opts.get('update_embedding', False)
        actFunc = loadActivationFunction(sum_desc_opts.get('activation', 'relu'))
        batchNorm = sum_desc_opts.get('batch_normalization', False)
        dropout = sum_desc_opts.get('dropout', 0.0)

        sumDescEncoder = TextCNN(windowSizes, nFilters, embedding, updateEmb, actFunc, batchNorm, dropout)
        encoders.append(sumDescEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, max(windowSizes)))

    elif sum_desc_opts['encoder_type'] == 'cnn+dense':
        windowSizes = sum_desc_opts.get('window_sizes', [3])
        nFilters = sum_desc_opts.get('nfilters', 100)
        updateEmb = sum_desc_opts.get('update_embedding', False)
        actFunc = loadActivationFunction(sum_desc_opts.get('activation', 'relu'))
        batchNorm = sum_desc_opts.get('batch_normalization', False)
        dropout = sum_desc_opts.get('dropout', 0.0)
        hiddenSizes = sum_desc_opts.get('hidden_sizes')
        hiddenAct = loadActivationClass(sum_desc_opts.get('hidden_act'))
        hiddenDropout = sum_desc_opts.get('hidden_dropout')
        batchLast = sum_desc_opts.get("bn_last_layer", False)

        cnnEnc = TextCNN(windowSizes, nFilters, embedding, updateEmb, actFunc, batchNorm, dropout)
        sumDescEncoder = MultilayerDense(cnnEnc, hiddenSizes, hiddenAct, batchNorm, batchLast, hiddenDropout)
        encoders.append(sumDescEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, max(windowSizes)))
    elif sum_desc_opts['encoder_type'] == 'word_mean':
        standardization = sum_desc_opts.get('standardization', False)
        dropout = sum_desc_opts.get('dropout', 0.0)
        updateEmb = sum_desc_opts.get('update_embedding', False)
        batch_normalization = sum_desc_opts.get('update_embedding', False)
        hiddenSize = sum_desc_opts.get('hidden_size')

        sumDescEncoder = WordMean(embedding, updateEmb, hiddenSize, standardization, dropout, batch_normalization)

        encoders.append(sumDescEncoder)
        inputHandlers.append(RNNInputHandler(paddingId))
    else:
        raise ArgumentError(
            "Encoder type of summary and description is invalid (%s). You should choose one of these: cnn" %
            sum_desc_opts['encoder_type'])


def processSumParam(sumOpts, bugReportDatabase, inputHandlers, preprocessors, encoders, databasePath, cacheFolder, logger, paddingSym):
    # Use summary and description (concatenated) to address this problem
    logger.info("Using Summary information.")
    # Loading word embedding
    lexicon, embedding = load_embedding(sumOpts, paddingSym)
    logger.info("Lexicon size: %d" % (lexicon.getLen()))
    logger.info("Word Embedding size: %d" % (embedding.getEmbeddingSize()))
    paddingId = lexicon.getLexiconIndex(paddingSym)
    # Loading Filters
    filters = loadFilters(sumOpts['filters'])
    # Tokenizer
    if sumOpts['tokenizer'] == 'default':
        logger.info("Use default tokenizer to tokenize summary information")
        tokenizer = MultiLineTokenizer()
    elif sumOpts['tokenizer'] == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            sumOpts['tokenizer'])
    arguments = (
        databasePath, sumOpts['word_embedding'], str(sumOpts['lexicon']),
        ' '.join(sorted([fil.__class__.__name__ for fil in filters])),
        sumOpts['tokenizer'], "summary")

    summaryCache = PreprocessingCache(cacheFolder, arguments)
    sumDescPreprocessor = SummaryPreprocessor(lexicon, bugReportDatabase, filters, tokenizer, paddingId, summaryCache)

    preprocessors.append(sumDescPreprocessor)

    if sumOpts['encoder_type'] == 'rnn':
        rnnType = sumOpts.get('rnn_type')
        hiddenSize = sumOpts.get('hidden_size')
        bidirectional = sumOpts.get('bidirectional', False)
        numLayers = sumOpts.get('num_layers', 1)
        dropout = sumOpts.get('dropout', 0.0)
        updateEmb = sumOpts.get('update_embedding', False)
        fixedOpt = sumOpts.get('fixed_opt', False)

        sumRNN = SortedRNNEncoder(rnnType, embedding, hiddenSize, numLayers, bidirectional, updateEmb, dropout)
        if fixedOpt == 'self_att':
            att = SelfAttention(sumRNN.getOutputSize(), sumOpts['self_att_hidden'], sumOpts['n_hops'])
            summaryEncoder = RNN_Self_Attention(sumRNN, att, paddingId, dropout)
        else:
            summaryEncoder = RNNFixedOuput(sumRNN, fixedOpt, dropout)

        encoders.append(summaryEncoder)
        inputHandlers.append(RNNInputHandler(paddingId))
    elif sumOpts['encoder_type'] == 'cnn':
        windowSizes = sumOpts.get('window_sizes', [3])
        nFilters = sumOpts.get('nfilters', 100)
        updateEmb = sumOpts.get('update_embedding', False)
        actFunc = loadActivationFunction(sumOpts.get('activation', 'relu'))
        batchNorm = sumOpts.get('batch_normalization', False)
        dropout = sumOpts.get('dropout', 0.0)

        summaryEncoder = TextCNN(windowSizes, nFilters, embedding, updateEmb, actFunc, batchNorm, dropout)
        encoders.append(summaryEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, max(windowSizes)))
    elif sumOpts['encoder_type'] == 'cnn+dense':
        windowSizes = sumOpts.get('window_sizes', [3])
        nFilters = sumOpts.get('nfilters', 100)
        updateEmb = sumOpts.get('update_embedding', False)
        actFunc = loadActivationFunction(sumOpts.get('activation', 'relu'))
        batchNorm = sumOpts.get('batch_normalization', False)
        dropout = sumOpts.get('dropout', 0.0)
        hiddenSizes = sumOpts.get('hidden_sizes')
        hiddenAct = loadActivationClass(sumOpts.get('hidden_act'))
        hiddenDropout = sumOpts.get('hidden_dropout')
        batchLast = sumOpts.get("bn_last_layer", False)

        cnnEnc = TextCNN(windowSizes, nFilters, embedding, updateEmb, actFunc, batchNorm, dropout)
        summaryEncoder = MultilayerDense(cnnEnc, hiddenSizes, hiddenAct, batchNorm, batchLast, hiddenDropout)
        encoders.append(summaryEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, max(windowSizes)))
    elif sumOpts['encoder_type'] == 'dense+self_att':
        dropout = sumOpts.get('dropout', 0.0)
        hiddenSize = sumOpts.get('hidden_size')
        self_att_hidden = sumOpts['self_att_hidden']
        n_hops = sumOpts['n_hops']
        updateEmb = sumOpts.get('update_embedding', False)

        summaryEncoder = Dense_Self_Attention(embedding, hiddenSize, self_att_hidden, n_hops, paddingId, updateEmb, dropout=dropout)
        encoders.append(summaryEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, -1))
    elif sumOpts['encoder_type'] == 'word_mean':
        standardization = sumOpts.get('standardization', False)
        dropout = sumOpts.get('dropout', 0.0)
        updateEmb = sumOpts.get('update_embedding', False)
        batch_normalization = sumOpts.get('update_embedding', False)
        hiddenSize = sumOpts.get('hidden_size')

        summaryEncoder = WordMean( embedding, updateEmb, hiddenSize, standardization, dropout, batch_normalization)

        encoders.append(summaryEncoder)
        inputHandlers.append(RNNInputHandler(paddingId))
    else:
        raise ArgumentError(
            "Encoder type of summary and description is invalid (%s). You should choose one of these: cnn" %
            sumOpts['encoder_type'])


def processDescriptionParam(descOpts, bugReportDatabase, inputHandlers, preprocessors, encoders, databasePath, cacheFolder, logger, paddingSym):
    # Use summary and description (concatenated) to address this problem
    logger.info("Using Description information.")
    # Loading word embedding

    lexicon, embedding = load_embedding(descOpts, paddingSym)
    logger.info("Lexicon size: %d" % (lexicon.getLen()))
    logger.info("Word Embedding size: %d" % (embedding.getEmbeddingSize()))
    paddingId = lexicon.getLexiconIndex(paddingSym)
    # Loading Filters
    filters = loadFilters(descOpts['filters'])
    # Tokenizer
    if descOpts['tokenizer'] == 'default':
        logger.info("Use default tokenizer to tokenize summary information")
        tokenizer = MultiLineTokenizer()
    elif descOpts['tokenizer'] == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            descOpts['tokenizer'])

    arguments = (
        databasePath, descOpts['word_embedding'], str(descOpts['lexicon']),
        ' '.join(sorted([fil.__class__.__name__ for fil in filters])),
        descOpts['tokenizer'], "description")

    descCache = PreprocessingCache(cacheFolder, arguments)
    descPreprocessor = DescriptionPreprocessor(lexicon, bugReportDatabase, filters, tokenizer, paddingId, descCache)
    preprocessors.append(descPreprocessor)

    if descOpts['encoder_type'] == 'rnn':
        rnnType = descOpts.get('rnn_type')
        hiddenSize = descOpts.get('hidden_size')
        bidirectional = descOpts.get('bidirectional', False)
        numLayers = descOpts.get('num_layers', 1)
        dropout = descOpts.get('dropout', 0.0)
        updateEmb = descOpts.get('update_embedding', False)
        fixedOpt = descOpts.get('fixed_opt', False)

        descRNN = SortedRNNEncoder(rnnType, embedding, hiddenSize, numLayers, bidirectional, updateEmb, dropout)

        if fixedOpt == 'self_att':
            att = SelfAttention(descRNN.getOutputSize(), descOpts['self_att_hidden'], descOpts['n_hops'])
            descEncoder = RNN_Self_Attention(descRNN, att, paddingId, dropout)
        else:
            descEncoder = RNNFixedOuput(descRNN, fixedOpt, dropout)

        encoders.append(descEncoder)
        inputHandlers.append(RNNInputHandler(paddingId))
    elif descOpts['encoder_type'] == 'cnn':
        windowSizes = descOpts.get('window_sizes', [3])
        nFilters = descOpts.get('nfilters', 100)
        updateEmb = descOpts.get('update_embedding', False)
        actFunc = loadActivationFunction(descOpts.get('activation', 'relu'))
        batchNorm = descOpts.get('batch_normalization', False)
        dropout = descOpts.get('dropout', 0.0)

        descEncoder = TextCNN(windowSizes, nFilters, embedding, updateEmb, actFunc, batchNorm, dropout)
        encoders.append(descEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, max(windowSizes)))
    elif descOpts['encoder_type'] == 'cnn+dense':
        windowSizes = descOpts.get('window_sizes', [3])
        nFilters = descOpts.get('nfilters', 100)
        updateEmb = descOpts.get('update_embedding', False)
        actFunc = loadActivationFunction(descOpts.get('activation', 'relu'))
        batchNorm = descOpts.get('batch_normalization', False)
        dropout = descOpts.get('dropout', 0.0)
        hiddenSizes = descOpts.get('hidden_sizes')
        hiddenAct = loadActivationClass(descOpts.get('hidden_act'))
        hiddenDropout = descOpts.get('hidden_dropout')
        batchLast = descOpts.get("bn_last_layer", False)

        cnnEnc = TextCNN(windowSizes, nFilters, embedding, updateEmb, actFunc, batchNorm, dropout)
        descEncoder = MultilayerDense(cnnEnc, hiddenSizes, hiddenAct, batchNorm, batchLast, hiddenDropout)
        encoders.append(descEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, max(windowSizes)))
    elif descOpts['encoder_type'] == 'dense+self_att':
        dropout = descOpts.get('dropout', 0.0)
        hiddenSize = descOpts.get('hidden_size')
        self_att_hidden = descOpts['self_att_hidden']
        n_hops = descOpts['n_hops']
        updateEmb = descOpts.get('update_embedding', False)

        descEncoder = Dense_Self_Attention(embedding, hiddenSize, self_att_hidden, n_hops, paddingId, updateEmb, dropout=dropout)
        encoders.append(descEncoder)
        inputHandlers.append(TextCNNInputHandler(paddingId, -1))
    elif descOpts['encoder_type'] == 'word_mean':
        standardization = descOpts.get('standardization', False)
        dropout = descOpts.get('dropout', 0.0)
        updateEmb = descOpts.get('update_embedding', False)
        batch_normalization = descOpts.get('update_embedding', False)
        hiddenSize = descOpts.get('hidden_size')

        descEncoder = WordMean( embedding, updateEmb, hiddenSize, standardization, dropout, batch_normalization)

        encoders.append(descEncoder)
        inputHandlers.append(RNNInputHandler(paddingId))
    else:
        raise ArgumentError(
            "Encoder type of summary and description is invalid (%s). You should choose one of these: cnn" %
            descOpts['encoder_type'])


def processCategoricalParam(categoricalOpt, bugReportDatabase, inputHandlers, preprocessors, encoders, logger):
    logger.info("Using Categorical Information.")
    categoricalPreprocessors, categoricalLexicons = createCategoricalPreprocessorAndLexicons(
        categoricalOpt['lexicons'], bugReportDatabase)

    handler = BasicInputHandler(transpose_input=True)

    if inputHandlers is not None:
        inputHandlers.append(handler)

    if preprocessors is not None:
        preprocessors.append(categoricalPreprocessors)


    # Create model
    embeddingSize = categoricalOpt.get('emb_size', 20)
    hiddenSizes = categoricalOpt.get('hidden_sizes')
    batchNorm = categoricalOpt.get('batch_normalization', False)
    layerNorm = categoricalOpt.get('layer_norm', False)
    dropout = categoricalOpt.get('dropout', 0.0)
    actFunc = loadActivationClass(categoricalOpt.get('activation'))
    bnLastLayer = categoricalOpt.get('bn_last_layer', False)
    categoricalEncoder = CategoricalEncoder(categoricalLexicons, embeddingSize, hiddenSizes, actFunc, batchNorm, bnLastLayer,dropout,layerNorm)

    if encoders  is not None:
        encoders.append(categoricalEncoder)
    
    return categoricalEncoder, handler, categoricalPreprocessors
