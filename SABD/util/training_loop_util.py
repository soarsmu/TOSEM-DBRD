import logging

import numpy as np
from time import time

from torch import Tensor

from metrics.metric import cmAccuracy, cmPrecision, cmRecall
from metrics.ranking import RecallRate, RankingResultFile, generateRecommendationList, SunRanking


def logConfusionMatrix(expRun, logger, metricName, metricValue, epoch):
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
    })

    expRun.log_scalar(metricName + "_acc", acc, step=epoch)
    expRun.log_scalar(metricName + "_prec", prec[1], step=epoch)
    expRun.log_scalar(metricName + "_recall", recall[1], step=epoch)
    expRun.log_scalar(metricName + "_f1", f1[1], step=epoch)


def logMetrics(expRun, logger, metrics, epoch):
    for metricName, metricValue in metrics.items():
        if isinstance(metricValue, Tensor):
            continue

        if isinstance(metricValue, dict):
            o = {'type': 'metric', 'label': metricName, 'epoch': epoch}
            o.update(metricValue)
            logger.info(o)
        else:
            logger.info({
                    'type': 'metric', 
                    'label': metricName, 
                    'value': metricValue, 
                    'epoch': epoch
                }
            )

        expRun.log_scalar(metricName, metricValue, step=epoch)


def logRankingResult(expRun, logger, rankingClass, rankingScorer, bugReportDatabase, rankingResultFilePath, epoch, label=None, groupByMaster=True, recommendationListfn=generateRecommendationList):
    recallRateMetric = RecallRate(bugReportDatabase, groupByMaster = groupByMaster)
    rankingResultFile = RankingResultFile(rankingResultFilePath, bugReportDatabase) if rankingResultFilePath else None

    start_time = time()
    rankingScorer.pregenerateBugEmbedding(rankingClass.getAllBugs())

    rankingScorer.reset()

    positions = []

    for i, duplicateBugId in enumerate(rankingClass.getDuplicateBugs()):
        candidates = rankingClass.getCandidateList(duplicateBugId)

        if i > 0 and i % 500 == 0:
            logger.info('RR calculation - {} duplicate reports were processed'.format(i))

        if len(candidates) == 0:
            if isinstance(rankingClass, SunRanking):
                # If the window of days is too small to contain a duplicate bug, so this can happen.
                logging.getLogger().warning("Bug {} has 0 candidates!".format(duplicateBugId))
            else:
                # This shouldn't happen with the other methodologies
                raise Exception("Bug {} has 0 candidates!".format(duplicateBugId))

            recommendation = candidates
        else:
            recommendation = recommendationListfn(duplicateBugId, candidates, rankingScorer)

        # Update the metrics
        pos, correct_cand = recallRateMetric.update(duplicateBugId, recommendation)

        positions.append(pos)

        if rankingResultFile:
            rankingResultFile.update(duplicateBugId, recommendation, pos, correct_cand)

    recallRateResult = recallRateMetric.compute()

    end_time = time()
    nDupBugs = len(rankingClass.getDuplicateBugs())
    duration = end_time - start_time

    label = "_%s" % label if label and len(label) > 0 else ""

    logger.info(
        '[{}] Throughput: {} bugs per second (bugs={} ,seconds={})'.format(label, (nDupBugs / duration), nDupBugs, duration))

    for k, rate in recallRateResult.items():
        hit = recallRateMetric.hitsPerK[k]
        total = recallRateMetric.nDuplicate
        logger.info({
                'type': "metric", 
                'label': 'recall_rate%s' % (label), 
                'k': k, 
                'rate': rate, 
                'hit': hit,
                'total': total,
                'epoch': epoch
            }
        )

        expRun.log_scalar('recall_rate%s_%d' % (label, k), rate, step=epoch)

    valid_queries = np.asarray(positions)
    MAP_sum = (1 / valid_queries).sum()
    MAP = MAP_sum / valid_queries.shape[0]
    expRun.log_scalar('MAP', MAP, step=epoch)

    logger.info({
            'type': "metric", 
            'label': 'MAP', 
            'value': MAP, 
            'sum': MAP_sum, 
            'total': valid_queries.shape[0],
            'epoch': epoch
        }
    )

    logger.info('{}'.format(positions))
