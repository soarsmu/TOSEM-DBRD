"""
This script implements a siamese neural network which extract the information from each bug report of a pair.
The extracted features from each pair is used to calculate the similarity of them or probability of being duplicate.
"""

import logging
from argparse import ArgumentError
from datetime import datetime
import ignite
import numpy as np
import torch
from ignite.engine import Events, Engine
from sacred import Experiment
from torch import optim
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import StepLR, ExponentialLR, LambdaLR
from torch.utils.data import DataLoader

import sys
sys.path.append('./')


from data.bug_dataset import BugDataset
from data.bug_report_database import BugReportDatabase
from data.collate import PairBugCollate
from data.dataset import PairBugDatasetReader
from data.preprocessing import PreprocessorList
from example_generator.offline_pair_generation import NonNegativeRandomGenerator
from metrics.metric import AverageLoss, MeanScoreDistance, ConfusionMatrix, cmAccuracy, cmPrecision, cmRecall, \
    PredictionCache, LossWrapper
from metrics.ranking import SharedEncoderNNScorer, \
    SunRanking
from model.loss import CosineLoss, NeculoiuLoss
from model.siamese import ProbabilityPairNN, CosinePairNN
from util.jsontools import JsonLogFormatter
from util.siamese_util import processSumDescParam, processSumParam, processCategoricalParam, processDescriptionParam
from util.training_loop_util import logMetrics, logRankingResult, logConfusionMatrix

ex = Experiment("siamese_pairs")


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
    dropout = None
    batch_size = 16
    ranking_batch_size = 256
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
    sum_desc = {
        "word_embedding": None,
        "lexicon": None,
        "tokenizer": None,
        "filters": ["TransformLowerCaseFilter"],
        "encoder_type": "cnn",
        "window_sizes": [3],
        "nfilters": 100,
        "update_embedding": False,
        "activation": "relu",
        "batch_normalization": False,
        "dropout": 0.0,
        "hidden_sizes": None,
        "hidden_size": 0.0,
        "hidden_act": None,
        "hidden_dropout": 0.0,
        "bn_last_layer": False,
        "standardization": False
    }
    summary = {
        "word_embedding": None,
        "lexicon": None,
        "encoder_type": None,
        "tokenizer": None,
        "filters": ["TransformLowerCaseFilter"],
        'rnn_type': None,
        'hidden_size': 100,
        "nfilters": 100,
        'bidirectional': False,
        'num_layers': 1,
        'dropout': 0.0,
        'update_embedding': False,
        'fixed_opt': 'mean',
        "activation": "relu",
        "batch_normalization": False,
        "hidden_sizes": None,
        "hidden_act": None,
        "hidden_dropout": 0.0,
        "bn_last_layer": False,
        "self_att_hidden": 100,
        "n_hops": 20,
        "standardization": False
    }

    description = {
        "word_embedding": None,
        "lexicon": None,
        "encoder_type": None,
        "tokenizer": None,
        "filters": ["TransformLowerCaseFilter"],
        'rnn_type': None,
        'hidden_size': 100,
        "nfilters": 100,
        'bidirectional': False,
        'num_layers': 1,
        'dropout': 0.0,
        'update_embedding': False,
        'fixed_opt': 'mean',
        "activation": "relu",
        "batch_normalization": False,
        "hidden_sizes": None,
        "hidden_act": None,
        "hidden_dropout": 0.0,
        "bn_last_layer": False,
        "window_sizes": [3],
        "self_att_hidden": 100,
        "n_hops": 20,
        "standardization": False
    }
    categorical = {
        "lexicons": None,
        "bn_last_layer": False,
        "emb_size": 20,
        "hidden_sizes": None,
        "dropout": 0.0,
        "activation": None,
        "batch_normalization": False
    }
    classifier = {
        "type": "binary",
        "without_embedding": False,
        "batch_normalization": False,
        "dropout": 0,
        "hidden_sizes": [100, 200],
        "margin": 0,
        "loss": None
    }
    recall_estimation_train = None
    recall_estimation = None
    rr_val_epoch = 1
    rr_train_epoch = 5
    random_switch = False
    ranking_result_file = None
    optimizer = "adam"
    momentum = 0.9
    lr_scheduler = {
        "type": "linear",
        "decay": 1,
        "step_size": 1
    }
    save = None
    load = None
    pair_test_dataset = None

    recall_rate = {
        'type': 'none',  # 3 options: none, sun2011 and deshmukh
        'dataset': None,
        'result_file': None,
        'group_by_master': True,
        'window': None  # only compare bug that are in this interval of days

        # File where we store the position of each duplicate bug in the list, the first 30 top reports,
    }


@ex.automain
def main(_run, _config, _seed, _log):
    # Setting logger
    args = _config
    logger = _log
    PROJECT = args['database_name']

    file_handler = logging.FileHandler('./result_log/pairs_{}_{}.log'.format(PROJECT, datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

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
    importantParameters = ['summary', 'description', 'sum_desc', 'classifier', 'categorical']
    parametersToSave = dict([(parName, args[parName]) for parName in importantParameters])

    if args['load'] is not None:
        mapLocation = (lambda storage, loc: storage.cuda()) if args['cuda'] else 'cpu'
        modelInfo = torch.load(args['load'], map_location=mapLocation)
        modelState = modelInfo['model']

        for paramName, paramValue in modelInfo['params'].items():
            args[paramName] = paramValue
    else:
        modelState = None

    """
    Set preprocessor that will pre-process the raw information from the bug reports.
    Each different information has a specific encoder(NN), preprocessor and input handler.
    """

    preprocessors = PreprocessorList()
    encoders = []
    inputHandlers = []
    globalDropout = args['dropout']

    databasePath = args['bug_database']

    sum_desc_opts = args['sum_desc']

    if sum_desc_opts is not None:
        if globalDropout:
            args['sum_desc']['dropout'] = globalDropout

        processSumDescParam(
            sum_desc_opts, 
            bugReportDatabase, 
            inputHandlers, 
            preprocessors, 
            encoders, 
            cacheFolder,
            databasePath, 
            logger, 
            paddingSym
        )

    sumOpts = args.get("summary")

    if sumOpts is not None:
        if globalDropout:
            args['summary']['dropout'] = globalDropout

        processSumParam(
            sumOpts,
            bugReportDatabase, 
            inputHandlers, 
            preprocessors, 
            encoders, 
            databasePath, 
            cacheFolder,
            logger, 
            paddingSym
        )

    descOpts = args.get("description")

    if descOpts is not None:
        if globalDropout:
            args['description']['dropout'] = globalDropout

        processDescriptionParam(
            descOpts, 
            bugReportDatabase, 
            inputHandlers, 
            preprocessors, 
            encoders, 
            databasePath,
            cacheFolder,
            logger, 
            paddingSym
        )

    categoricalOpt = args.get('categorical')

    if categoricalOpt is not None and len(categoricalOpt) != 0:
        if globalDropout:
            args['categorical']['dropout'] = globalDropout

        processCategoricalParam(categoricalOpt, bugReportDatabase, inputHandlers, preprocessors, encoders, logger)

    """
    Set the final classifier and the loss. Load the classifier if this argument was set.
    """
    classifierOpts = args['classifier']
    classifierType = classifierOpts['type']
    labelDType = None

    if globalDropout:
        args['classifier']['dropout'] = globalDropout

    if classifierType == 'binary':
        withoutBugEmbedding = classifierOpts.get('without_embedding', False)
        batchNorm = classifierOpts.get('batch_normalization', True)
        dropout = classifierOpts.get('dropout', 0.0)
        hiddenSizes = classifierOpts.get('hidden_sizes', [100])
        model = ProbabilityPairNN(encoders, withoutBugEmbedding, hiddenSizes, batchNorm, dropout)
        lossFn = NLLLoss()
        lossNoReduction = NLLLoss(reduction='none')

        labelDType = torch.int64

        logger.info("Using NLLLoss")
    elif classifierType == 'cosine':
        model = CosinePairNN(encoders)
        margin = classifierOpts.get('margin', 0.0)

        if classifierOpts['loss'] == 'cosine_loss':
            lossFn = CosineLoss(margin)
            lossNoReduction = CosineLoss(margin, reduction='none')
            labelDType = torch.float32
            logger.info("Using Cosine Embeding Loss: margin={}".format(margin))
        elif classifierOpts['loss'] == 'neculoiu_loss':
            lossFn = NeculoiuLoss(margin)
            lossNoReduction = NeculoiuLoss(margin, reduction='none')
            labelDType = torch.float32
            logger.info("Using Neculoiu Loss: margin={}".format(margin))

    model.to(device)

    if modelState:
        model.load_state_dict(modelState)

    """
    Loading the training and validation. Also, it sets how the negative example will be generated.
    """
    pairCollate = PairBugCollate(inputHandlers, labelDType)

    # load training
    if args.get('pairs_training'):
        negativePairGenOpt = args.get('neg_pair_generator',)
        pairsTrainingFile = args.get('pairs_training')
        randomAnchor = negativePairGenOpt['random_anchor']

        offlineGeneration = not (negativePairGenOpt is None or negativePairGenOpt['type'] == 'none')

        if not offlineGeneration:
            logger.info("Not generate dynamically the negative examples.")
            pairTrainingReader = PairBugDatasetReader(
                pairsTrainingFile, 
                preprocessors, 
                randomInvertPair=args['random_switch']
            )
        else:
            pairGenType = negativePairGenOpt['type']
            masterIdByBugId = bugReportDatabase.getMasterIdByBugId()

            if pairGenType == 'non_negative':
                logger.info("Non Negative Pair Generator")
                trainingDataset = BugDataset(negativePairGenOpt['training'])
                bugIds = trainingDataset.bugIds

                logger.info(
                    "Using the following dataset to generate negative examples: %s. Number of bugs in the training: %d" % (
                        trainingDataset.info, len(bugIds))
                )

                negativePairGenerator = NonNegativeRandomGenerator(
                    preprocessors, 
                    pairCollate, 
                    negativePairGenOpt['rate'], 
                    bugIds, 
                    masterIdByBugId, 
                    negativePairGenOpt['n_tries'],
                    device, 
                    randomAnchor=randomAnchor
                )

            else:
                raise ArgumentError(
                    "Offline generator is invalid (%s). You should choose one of these: random, hard and pre" %
                    pairGenType
                )

            pairTrainingReader = PairBugDatasetReader(pairsTrainingFile, preprocessors, negativePairGenerator, randomInvertPair=args['random_switch'])

        trainingLoader = DataLoader(pairTrainingReader, batch_size=batchSize, collate_fn=pairCollate.collate, shuffle=True)
        logger.info("Training size: %s" % (len(trainingLoader.dataset)))

    # load validation
    if args.get('pairs_validation'):
        pairValidationReader = PairBugDatasetReader(args.get('pairs_validation'), preprocessors)
        validationLoader = DataLoader(pairValidationReader, batch_size=batchSize, collate_fn=pairCollate.collate)

        logger.info("Validation size: %s" % (len(validationLoader.dataset)))
    else:
        validationLoader = None

    """
    Training and evaluate the model. 
    """
    optimizer_opt = args.get('optimizer', 'adam')

    if optimizer_opt == 'sgd':
        logger.info('SGD')
        optimizer = optim.SGD(model.parameters(), lr=args['lr'], weight_decay=args['l2'], momentum=args['momentum'])
    elif optimizer_opt == 'adam':
        logger.info('Adam')
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2'])

    # Recall rate
    rankingScorer = SharedEncoderNNScorer(preprocessors, inputHandlers, model, device, args['ranking_batch_size'])

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

    def scoreDistanceTrans(output):
        if len(output) == 3:
            _, y_pred, y = output
        else:
            y_pred, y = output

        if isinstance(lossFn, NLLLoss):
            return torch.exp(y_pred[:, 1]), y
        elif isinstance(lossFn, CosineLoss):
            return y_pred, (y * 2) - 1

    # Set training functions
    def trainingIteration(engine, batch):
        model.train()
        optimizer.zero_grad()
        (bug1, bug2), y = pairCollate.to(batch, device)
        output = model(bug1, bug2)
        loss = lossFn(output, y)
        loss.backward()
        optimizer.step()
        return loss, output, y

    trainer = Engine(trainingIteration)
    negTarget = 0.0 if isinstance(lossFn, NLLLoss) else -1.0

    trainingMetrics = {
        'training_loss': AverageLoss(lossFn),
        'training_dist_target': MeanScoreDistance(negTarget=negTarget, output_transform=scoreDistanceTrans),
        'training_confusion_matrix': ConfusionMatrix(2, output_transform=lambda x: (x[1], x[2])), 
    }

    # Add metrics to trainer
    for name, metric in trainingMetrics.items():
        metric.attach(trainer, name)

    # Set validation functions
    def validationIteration(engine, batch):
        model.eval()
        with torch.no_grad():
            (bug1, bug2), y = pairCollate.to(batch, device)
            y_pred = model(bug1, bug2)
            return y_pred, y

    validationMetrics = {
        'validation_loss': LossWrapper(lossFn),
        'validation_dist_target': MeanScoreDistance(negTarget=negTarget, output_transform=scoreDistanceTrans), 'validation_confusion_matrix': ConfusionMatrix(2), 
    }
    evaluator = Engine(validationIteration)

    # Add metrics to evaluator
    for name, metric in validationMetrics.items():
        metric.attach(evaluator, name)

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

        logConfusionMatrix(_run, logger, 'training_confusion_matrix', engine.state.metrics['training_confusion_matrix'], epoch)
        logMetrics(_run, logger, engine.state.metrics, epoch)

        # Evaluate Training
        if validationLoader:
            evaluator.run(validationLoader)
            logConfusionMatrix(
                _run, 
                logger, 
                'validation_confusion_matrix',
                evaluator.state.metrics['validation_confusion_matrix'], 
                epoch
            )

            logMetrics(_run, logger, evaluator.state.metrics, epoch)

        if offlineGeneration:
            pairTrainingReader.sampleNewNegExamples(model, lossNoReduction)

        if args.get('save'):
            modelInfo = {'model': model.state_dict(),
                        'params': parametersToSave}

            logger.info("==> Saving Model: %s" % args['save'])
            torch.save(modelInfo, args['save'])

    if args.get('pairs_training'):
        trainer.run(trainingLoader, max_epochs=args['epochs'])
    elif args.get('pairs_validation'):
        # Evaluate Training
        evaluator.run(trainingLoader)
        logMetrics(logger, evaluator.state.metrics)

    # Test Dataset (accuracy, recall, precision, F1)
    pair_test_dataset = args.get('pair_test_dataset')

    if pair_test_dataset is not None and len(pair_test_dataset) > 0:
        pairTestReader = PairBugDatasetReader(
            pair_test_dataset, 
            preprocessors
        )
        testLoader = DataLoader(pairTestReader, batch_size=batchSize, collate_fn=pairCollate.collate)

        logger.info("Test size: %s" % (len(testLoader.dataset)))

        testMetrics = {
            'test_accuracy': ignite.metrics.Accuracy(),
            'test_precision': ignite.metrics.Precision(),
            'test_recall': ignite.metrics.Recall(),
            'test_confusion_matrix': ConfusionMatrix(2),
            'test_predictions': PredictionCache(), 
        }
        test_evaluator = Engine(validationIteration)

        # Add metrics to evaluator
        for name, metric in testMetrics.items():
            metric.attach(test_evaluator, name)

        test_evaluator.run(testLoader)

        for metricName, metricValue in test_evaluator.state.metrics.items():
            metric = testMetrics[metricName]

            if isinstance(metric, ignite.metrics.Accuracy):
                logger.info({
                        'type': 'metric', 
                        'label': metricName, 
                        'value': metricValue, 
                        'epoch': None,
                        'correct': metric._num_correct, 
                        'total': metric._num_examples
                    }
                )
                _run.log_scalar(metricName, metricValue)
            elif isinstance(metric, (ignite.metrics.Precision, ignite.metrics.Recall)):
                logger.info({
                        'type': 'metric', 
                        'label': metricName, 
                        'value': np.float(metricValue.cpu().numpy()[1]),
                        'epoch': None,
                        'tp': metric._true_positives.cpu().numpy().tolist(),
                        'total_positive': metric._positives.cpu().numpy().tolist()
                    }
                )
                _run.log_scalar(metricName, metricValue[1])
            elif isinstance(metric, ConfusionMatrix):
                acc = cmAccuracy(metricValue)
                prec = cmPrecision(metricValue, False)
                recall = cmRecall(metricValue, False)
                f1 = 2 * (prec * recall) / (prec + recall + 1e-15)

                logger.info({
                        'type': 'metric', 'label': metricName,
                        'accuracy': np.float(acc),
                        'precision': prec.cpu().numpy().tolist(),
                        'recall': recall.cpu().numpy().tolist(),
                        'f1': f1.cpu().numpy().tolist(),
                        'confusion_matrix': metricValue.cpu().numpy().tolist(),
                        'epoch': None
                    }
                )

                _run.log_scalar('test_f1', f1[1])
            elif isinstance(metric, PredictionCache):
                logger.info({
                        'type': 'metric', 
                        'label': metricName,
                        'predictions': metric.predictions
                    }
                )

    # Calculate recall rate
    recallRateOpt = args.get('recall_rate', {'type': 'none'})
    if recallRateOpt['type'] != 'none':
        if recallRateOpt['type'] == 'sun2011':
            logger.info("Calculating recall rate: {}".format(recallRateOpt['type']))
            recallRateDataset = BugDataset(recallRateOpt['dataset'])

            rankingClass = SunRanking(bugReportDatabase, recallRateDataset, recallRateOpt['window'])
            # We always group all bug reports by master in the results in the sun 2011 methodology
            group_by_master = True
        else:
            raise ArgumentError(
                "recall_rate.type is invalid (%s). You should choose one of these: step, exp and linear " %
                recallRateOpt['type'])

        logRankingResult(_run, logger, rankingClass, rankingScorer, bugReportDatabase, recallRateOpt["result_file"], 0, None, group_by_master)

        end_time = datetime.now()
        logger.info('It completed at: {}'.format(end_time))
        logger.info('Completed after: {}'.format(end_time - start_time))