import codecs
import logging
import os
from argparse import ArgumentError
from datetime import datetime

import torch
torch.manual_seed(941207)

import torch.nn.functional as F
import numpy as np
from ignite.engine import Events, Engine
from nltk import WhitespaceTokenizer
from sacred import Experiment
from torch import optim
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from torch.utils.data import DataLoader

import sys
sys.path.append('./')

from data.Embedding import Embedding
from data.Lexicon import Lexicon
from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.collate import PairBugCollate, TripletBugCollate
from data.dataset import PairBugDatasetReader, TripletBugDatasetReader
from data.input_handler import SABDInputHandler
from data.preprocessing import PreprocessingCache, MultiLineTokenizer, loadFilters, PreprocessorList, SABDEncoderPreprocessor
from example_generator.offline_pair_generation import NonNegativeRandomGenerator
from metrics.metric import AverageLoss, MeanScoreDistance, AccuracyWrapper, PrecisionWrapper, RecallWrapper, LossWrapper
from metrics.ranking import GeneralScorer, SunRanking, generateRecommendationList
from model.compare_aggregate import SABD
from model.loss import TripletLoss
from util.jsontools import JsonLogFormatter
from util.siamese_util import processCategoricalParam
from util.torch_util import thresholded_output_transform
from util.training_loop_util import logMetrics, logRankingResult

ex = Experiment("sabd")

logger = logging.getLogger()

ex.logger = logger


@ex.config
def cfg():
    # Set here all possible parameters; You have to change some parameters values.
    bug_database = None
    database_name = None
    epochs = 20
    lr = 0.001
    l2 = 0.0
    batch_size = 16
    ranking_batch_size = 256
    ranking_n_workers = 2
    cuda = True
    cache_folder = None
    pairs_training = None
    pairs_validation = None
    neg_pair_generator = {
        "type": "none",
        "training": None,
        "rate": 1,
        "pre_list_file": None,
        "k": 0,
        "n_tries": 0,
        "preselected_length": None,
        "random_anchor": True
    }

    loss = "bce"
    margin = 0
    categorical = {
        "lexicons": None,
        "bn_last_layer": False,
        "emb_size": 20,
        "hidden_sizes": None,
        "dropout": 0.0,
        "activation": None,
        "batch_normalization": False,
        "layer_norm": False
    }
    compare_aggregation = {
        "word_embedding": None,
        "norm_word_embedding": False,
        "lexicon": None,
        "sent_representation": None,
        "tokenizer": None,
        "bow": False,
        "frequency": False,
        "extractor": {
            'update_embedding': False,
            "filters": ["TransformLowerCaseFilter"],
            'hidden_size': 150,
            'dropout': 0.0,
            "layer_norm": False,
            "txt_field_emb_size": 5,
            "field_word_combination": "cat",
            "emb_dropout": 0.0,
            "scaled_attention": True,
            "model": "linear",
            "use_categorical": False,
            "bidirectional": True,
        },
        "matching": {
            "type": "full",  # full, partial and none
            "scaled_attention": True,
            "comparison_hidden_size": 100,
            'dropout': 0.0,
            'residual': True,
            "layer_norm": False,
            'attention_hidden_size': 200,
            "attention": "dot_product",
            "comparison_func": "submult+nn"
        },
        "aggregate": {
            "model": "cnn",
            "window": [3, 4, 5],
            "nfilters": 100,
            "hidden_size": 150,
            'dropout': 0.0,
            'bidirectional': True,
            'num_layers': 1,
            'self_att_hidden': 50,
            'layer_norm': False,
            'n_hops': 20
        },
        "classifier": {
            "hidden_size": [300],
            "output_act": "sigmoid",
            "hadamard_diff_textual": False,
            "hadamard_diff_categorical": False,
            "textual_hidden_layer": 0,
            "categorical_hidden_layer": 0,
            'categorical_dropout': 0.0,
            'dropout': 0.0,
            "only_candidate": False,
            'layer_norm': False,
            "batch_normalization": False,
        }
    }
    random_switch = False
    recall_estimation_train = None
    recall_estimation = None
    sample_size_rr_tr = 0  # Number of bug report in the training that will be sample to evaluate the recall rate
    sample_size_rr_val = 0  # Number of bug report in the validation that will be sample to evaluate the recall rate
    rr_val_epoch = 1
    rr_train_epoch = 5
    ranking_result_file = None
    optimizer = "adam"
    lr_scheduler = {
        "type": "linear",
        "decay": 1,
        "step_size": 1
    }
    save = None
    save_by_epoch = None
    load = None
    recall_rate = {
        'type': 'none',  # 3 options: none, sun2011 and deshmukh
        'dataset': None,
        'result_file': None,
        'group_by_master': True,
        'window': None  # only compare bug that are in this interval of days

        # File where we store the position of each duplicate bug in the list, the first 30 top reports,
    }
    pair_test_dataset = None
    rep = {
        'model': None,
        'input': None,
        'training': None,
        'rate': 1,
        'neg_training': 10000,
        'k': 1000
    }
    rr_scorer = "general"  # Options: general or optimized


@ex.automain
def main(_run, _config, _seed, _log):
    """

    :param _run:
    :param _config:
    :param _seed:
    :param _log:
    :return:
    """

    """
    Setting and loading parameters
    """
    # Setting logger
    args = _config
    logger = _log
    
    # PROJECT = os.path.split(args['bug_database'])[1].split('_')[0]
    PROJECT = args['database_name']
    os.makedirs('./result-logs/', exist_ok=True)
    file_handler = logging.FileHandler('./result-logs/sabd_{}_{}.log'.format(PROJECT, datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s', datefmt = '%F %A %T'))
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logHandler = logging.StreamHandler()
    formatter = JsonLogFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.info(args)

    start_time = datetime.now()    
    logger.info('It started at: %s' % start_time)

    torch.manual_seed(_seed)
    bugReportDatabase = BugReportDatabase.fromJson(args['bug_database'])
    paddingSym = "</s>"
    batchSize = args['batch_size']

    device = torch.device('cuda' if args['cuda'] else "cpu")

    if args['cuda']:
        logger.info("Turning CUDA on")
    else:
        logger.info("Turning CUDA off")

    # It is the folder where the preprocessed information will be stored.
    # cacheFolder = args['cache_folder']
    cacheFolder = None

    # Setting the parameter to save and loading parameters
    importantParameters = ['compare_aggregation', 'categorical']
    parametersToSave = dict([(parName, args[parName]) for parName in importantParameters])

    if args['load'] is not None:
        mapLocation = (lambda storage, loc: storage.cuda()) if args['cuda'] else 'cpu'
        modelInfo = torch.load(args['load'], map_location=mapLocation)
        modelState = modelInfo['model']

        for paramName, paramValue in modelInfo['params'].items():
            args[paramName] = paramValue
    else:
        modelState = None

    preprocessors = PreprocessorList()
    inputHandlers = []

    categoricalOpt = args.get('categorical')

    if categoricalOpt is not None and len(categoricalOpt) != 0:
        categoricalEncoder, _, _ = processCategoricalParam(categoricalOpt, bugReportDatabase, \
            inputHandlers, preprocessors, None, logger)
    else:
        categoricalEncoder = None

    compareAggOpt = args['compare_aggregation']
    databasePath = args['bug_database']

    # Loading word embedding
    if compareAggOpt["lexicon"]:
        emb = np.load(compareAggOpt["word_embedding"])

        lexicon = Lexicon(unknownSymbol=None)
        with codecs.open(compareAggOpt["lexicon"]) as f:
            for l in f:
                lexicon.put(l.strip())

        lexicon.setUnknown("UUUKNNN")
        paddingId = lexicon.getLexiconIndex(paddingSym)
        embedding = Embedding(lexicon, emb, paddingIdx=paddingId)

        logger.info("Lexicon size: %d" % (lexicon.getLen()))
        logger.info("Word Embedding size: %d" % (embedding.getEmbeddingSize()))
    else:
        embedding = None

    if compareAggOpt["norm_word_embedding"]:
        embedding.zscoreNormalization()

    # Tokenizer
    if compareAggOpt['tokenizer'] == 'default':
        logger.info("Use default tokenizer to tokenize summary information")
        tokenizer = MultiLineTokenizer()
    elif compareAggOpt['tokenizer'] == 'white_space':
        logger.info("Use white space tokenizer to tokenize summary information")
        tokenizer = WhitespaceTokenizer()
    else:
        raise ArgumentError(
            "Tokenizer value %s is invalid. You should choose one of these: default and white_space" %
            compareAggOpt['tokenizer']
        )

    # Preparing input handlers, preprocessors and cache
    minSeqSize = max(compareAggOpt['aggregate']["window"]) if compareAggOpt['aggregate']["model"] == "cnn" else -1
    bow = compareAggOpt.get('bow', False)
    freq = compareAggOpt.get('frequency', False) and bow

    logger.info("BoW={} and TF={}".format(bow, freq))

    if compareAggOpt['extractor'] is not None:
        # Use summary and description (concatenated) to address this problem
        logger.info("Using Summary and Description information.")
        # Loading Filters
        extractorFilters = loadFilters(compareAggOpt['extractor']['filters'])

        arguments = (databasePath, compareAggOpt['word_embedding'], str(compareAggOpt['lexicon']), \
            ' '.join(sorted([fil.__class__.__name__ for fil in extractorFilters])), compareAggOpt['tokenizer'], \
                str(bow), str(freq), SABDEncoderPreprocessor.__name__)

        inputHandlers.append(SABDInputHandler(paddingId, minSeqSize))
        extractorCache = PreprocessingCache(cacheFolder, arguments)

        extractorPreprocessor = SABDEncoderPreprocessor(
            lexicon, 
            bugReportDatabase, 
            extractorFilters, 
            tokenizer, 
            paddingId, 
            extractorCache
        )

        preprocessors.append(extractorPreprocessor)

    # Create model
    model = SABD(
        embedding, 
        categoricalEncoder, 
        compareAggOpt['extractor'], 
        compareAggOpt['matching'], 
        compareAggOpt['aggregate'], 
        compareAggOpt['classifier'], 
        freq
    )

    if args['loss'] == 'bce':
        logger.info("Using BCE Loss: margin={}".format(args['margin']))
        lossFn = BCELoss()
        lossNoReduction = BCELoss(reduction='none')
        cmp_collate = PairBugCollate(inputHandlers, torch.float32, unsqueeze_target=True)
    elif args['loss'] == 'triplet':
        logger.info("Using Triplet Loss: margin={}".format(args['margin']))
        lossFn = TripletLoss(args['margin'])
        lossNoReduction = TripletLoss(args['margin'], reduction='none')
        cmp_collate = TripletBugCollate(inputHandlers)

    model.to(device)

    if modelState:
        model.load_state_dict(modelState)

    """
    Loading the training and validation. Also, it sets how the negative example will be generated.
    """
    # load training
    if args.get('pairs_training'):
        negativePairGenOpt = args.get('neg_pair_generator')
        trainingFile = args.get('pairs_training')

        offlineGeneration = not (negativePairGenOpt is None or negativePairGenOpt['type'] == 'none')
        masterIdByBugId = bugReportDatabase.getMasterIdByBugId()
        randomAnchor = negativePairGenOpt['random_anchor']

        if not offlineGeneration:
            logger.info("Not generate dynamically the negative examples.")
            negativePairGenerator = None
        else:
            pairGenType = negativePairGenOpt['type']

            if pairGenType == 'non_negative':
                logger.info("Non Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds))
                )

                negativePairGenerator = NonNegativeRandomGenerator(preprocessors, cmp_collate, \
                    negativePairGenOpt['rate'], bugIds, masterIdByBugId, negativePairGenOpt['n_tries'], \
                        device, randomAnchor = randomAnchor)

            else:
                raise ArgumentError(
                    "Offline generator is invalid (%s). You should choose one of these: random, hard and pre" % pairGenType
                )

        if isinstance(lossFn, BCELoss):
            training_reader = PairBugDatasetReader(trainingFile, preprocessors, negativePairGenerator, \
                randomInvertPair = args['random_switch'])
        elif isinstance(lossFn, TripletLoss):
            training_reader = TripletBugDatasetReader(trainingFile, preprocessors, negativePairGenerator, \
                randomInvertPair = args['random_switch'])

        trainingLoader = DataLoader(
            training_reader, 
            batch_size = batchSize, 
            collate_fn = cmp_collate.collate, 
            shuffle = True
        )

        logger.info("Training size: %s" % (len(trainingLoader.dataset)))

    # load validation
    if args.get('pairs_validation'):
        if isinstance(lossFn, BCELoss):
            validation_reader = PairBugDatasetReader(
                args.get('pairs_validation'), 
                preprocessors
            )
        elif isinstance(lossFn, TripletLoss):
            validation_reader = TripletBugDatasetReader(
                args.get('pairs_validation'), 
                preprocessors
            )

        validationLoader = DataLoader(
            validation_reader, 
            batch_size = batchSize, 
            collate_fn = cmp_collate.collate
        )

        logger.info("Validation size: %s" % (len(validationLoader.dataset)))
    else:
        validationLoader = None

    """
    Training and evaluate the model. 
    """
    optimizer_opt = args.get('optimizer', 'adam')

    if optimizer_opt == 'sgd':
        logger.info('SGD')
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['l2'])
    elif optimizer_opt == 'adam':
        logger.info('Adam')
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2'])

    # Recall rate
    rankingScorer = GeneralScorer(
        model, 
        preprocessors, 
        device,
        PairBugCollate(inputHandlers, ignore_target=True),
        args['ranking_batch_size'], 
        args['ranking_n_workers']
    )

    # LR scheduler
    lrSchedulerOpt = args.get('lr_scheduler', None)

    if lrSchedulerOpt is None:
        logger.info("Scheduler: Constant")
        lrSched = None
    elif lrSchedulerOpt["type"] == 'step':
        logger.info("Scheduler: StepLR (step:%s, decay:%f)" % (lrSchedulerOpt["step_size"], args["decay"]))
        lrSched = StepLR(optimizer, lrSchedulerOpt["step_size"], lrSchedulerOpt["decay"])
    elif lrSchedulerOpt["type"] == 'exp':
        logger.info("Scheduler: ExponentialLR (decay:%f)" % (lrSchedulerOpt["decay"]))
        lrSched = ExponentialLR(optimizer, lrSchedulerOpt["decay"])
    elif lrSchedulerOpt["type"] == 'linear':
        logger.info("Scheduler: Divide by (1 + epoch * decay) ---- (decay:%f)" % (lrSchedulerOpt["decay"]))
        lrDecay = lrSchedulerOpt["decay"]
        lrSched = LambdaLR(optimizer, lambda epoch: 1 / (1.0 + epoch * lrDecay))
    else:
        raise ArgumentError(
            "LR Scheduler is invalid (%s). You should choose one of these: step, exp and linear " %
            pairGenType
        )

    # Set training functions
    def trainingIteration(engine, batch):
        engine.kk = 0
        model.train()

        optimizer.zero_grad()
        x, y = cmp_collate.to(batch, device)
        output = model(*x)
        loss = lossFn(output, y)
        loss.backward()
        optimizer.step()
        return loss, output, y

    def scoreDistanceTrans(output):
        if len(output) == 3:
            _, y_pred, y = output
        else:
            y_pred, y = output

        if lossFn == F.nll_loss:
            return torch.exp(y_pred[:, 1]), y
        elif isinstance(lossFn, (BCELoss)):
            return y_pred, y

    trainer = Engine(trainingIteration)
    trainingMetrics = {'training_loss': AverageLoss(lossFn)}

    if isinstance(lossFn, BCELoss):
        trainingMetrics['training_dist_target'] = MeanScoreDistance(output_transform = scoreDistanceTrans)
        trainingMetrics['training_acc'] = AccuracyWrapper(output_transform = thresholded_output_transform)
        trainingMetrics['training_precision'] = PrecisionWrapper(output_transform = thresholded_output_transform)
        trainingMetrics['training_recall'] = RecallWrapper(output_transform = thresholded_output_transform)
        # Add metrics to trainer
    for name, metric in trainingMetrics.items():
        metric.attach(trainer, name)

    # Set validation functions
    def validationIteration(engine, batch):
        if not hasattr(engine, 'kk'):
            engine.kk = 0

        model.eval()

        with torch.no_grad():
            x, y = cmp_collate.to(batch, device)
            y_pred = model(*x)

            return y_pred, y

    validationMetrics = {
        'validation_loss': LossWrapper(
            lossFn, 
            output_transform = lambda x: (x[0], x[0][0]) if x[1] is None else x
        )
    }

    if isinstance(lossFn, BCELoss):
        validationMetrics['validation_dist_target'] = MeanScoreDistance(output_transform = scoreDistanceTrans)
        validationMetrics['validation_acc'] = AccuracyWrapper(output_transform = thresholded_output_transform)
        validationMetrics['validation_precision'] = PrecisionWrapper(output_transform = thresholded_output_transform)
        validationMetrics['validation_recall'] = RecallWrapper(output_transform = thresholded_output_transform)

    evaluator = Engine(validationIteration)

    # Add metrics to evaluator
    for name, metric in validationMetrics.items():
        metric.attach(evaluator, name)

    # recommendation
    recommendation_fn = generateRecommendationList

    @trainer.on(Events.EPOCH_STARTED)
    def onStartEpoch(engine):
        epoch = engine.state.epoch
        logger.info("Epoch: %d" % epoch)

        if lrSched:
            lrSched.step()

        logger.info("LR: %s" % str(optimizer.param_groups[0]["lr"]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def onEndEpoch(engine):
        epoch = engine.state.epoch

        logMetrics(_run, logger, engine.state.metrics, epoch)

        # Evaluate Training
        if validationLoader:
            evaluator.run(validationLoader)
            logMetrics(_run, logger, evaluator.state.metrics, epoch)

        lastEpoch = args['epochs'] - epoch == 0

        if not lastEpoch:
            training_reader.sampleNewNegExamples(model, lossNoReduction)

        if args.get('save'):
            save_by_epoch = args['save_by_epoch']

            if save_by_epoch and epoch in save_by_epoch:
                file_name, file_extension = os.path.splitext(args['save'])
                file_path = file_name + '_epoch_{}'.format(epoch) + file_extension
            else:
                file_path = args['save']

            modelInfo = {
                'model': model.state_dict(),
                'params': parametersToSave
            }

            logger.info("==> Saving Model: %s" % file_path)
            torch.save(modelInfo, file_path)

    if args.get('pairs_training'):
        trainer.run(trainingLoader, max_epochs=args['epochs'])
    elif args.get('pairs_validation'):
        # Evaluate Training
        evaluator.run(validationLoader)
        logMetrics(_run, logger, evaluator.state.metrics, 0)

    # Calculate recall rate
    recallRateOpt = args.get('recall_rate', {'type': 'none'})
    if recallRateOpt['type'] != 'none':
        if recallRateOpt['type'] == 'sun2011':
            logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
            recallRateDataset = BugDataset(recallRateOpt['dataset'])

            rankingClass = SunRanking(
                bugReportDatabase, 
                recallRateDataset, 
                recallRateOpt['window']
            )
            # We always group all bug reports by master in the results in the sun 2011 methodology
            group_by_master = True
        else:
            raise ArgumentError(
                "recall_rate.type is invalid (%s). You should choose one of these: step, exp and linear " %
                recallRateOpt['type']
            )

        logRankingResult(_run, logger, rankingClass, rankingScorer, bugReportDatabase, \
            recallRateOpt["result_file"], 0, None, group_by_master, recommendationListfn = recommendation_fn)

        end_time = datetime.now()
        logger.info('It completed at: {}'.format(end_time))
        logger.info('Completed after: {}'.format(end_time - start_time))