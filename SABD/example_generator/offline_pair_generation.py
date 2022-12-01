import logging
import math
import random
import numpy as np
import torch


def sample_negative_report(masterIdByBugId, bugId, bugIdList):
    masterId = masterIdByBugId[bugId]
    current = bugIdList[random.randint(0, len(bugIdList) - 1)]
    currentMasterId = masterIdByBugId[current]

    if masterId == currentMasterId:
        return None

    return current


class BasicGenerator(object):

    def __init__(self, bugIdList, randomAnchor):
        self.possibleAnchors = None
        self.bugIdList = bugIdList
        self.randomAnchor = randomAnchor

    def setPossibleAnchors(self, pairsWithId):
        if self.randomAnchor:
            self.possibleAnchors = self.bugIdList
        else:
            if not self.possibleAnchors:
                possibleAnchors = set()

                for anchorId, candId, label in pairsWithId:
                    if label > 0:
                        possibleAnchors.add(anchorId)
                        possibleAnchors.add(candId)

                self.possibleAnchors = list(possibleAnchors)


class RandomGenerator(BasicGenerator):

    def __init__(self, preprocessor, collate, rate, bugIdList, masterIdByBugId, randomAnchor=True):
        super(RandomGenerator, self).__init__(bugIdList, randomAnchor)
        self.logger = logging.getLogger(__name__)
        self.preprocessor = preprocessor
        self.collate = collate
        self.rate = rate
        self.masterIdByBugId = masterIdByBugId

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negatives = []

        self.setPossibleAnchors(pairsWithId)

        for posPair in range(len(posPairs)):
            for _ in range(self.rate):
                anchorId = self.possibleAnchors[random.randint(0, len(self.possibleAnchors) - 1)]
                anchorIn = self.preprocessor.extract(anchorId)
                negId = self.sampleNegativeExample(anchorId)

                negatives.append((anchorIn, self.preprocessor.extract(negId), 0))

        return posPairs, negatives

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param tripletLoss:
        :param posPairs:
        :return:
        """
        triplets = []

        for (anchor, pos) in posPairs:
            anchorId, anchorIn = anchor
            posId, posIn = pos

            for _ in range(self.rate):
                negId = self.sampleNegativeExample(anchorId)

                triplets.append((anchorIn, posIn, self.preprocessor.extract(negId)))

        return triplets

    def sampleAnchor(self, k):
        return [self.possibleAnchors[random.randint(0, len(self.possibleAnchors) - 1)] for _ in range(k)]

    def sampleNegativeExample(self, bugId):
        masterId = self.masterIdByBugId[bugId]
        currentMasterId = masterId
        current = None

        while masterId == currentMasterId:
            current = self.bugIdList[random.randint(0, len(self.bugIdList) - 1)]
            currentMasterId = self.masterIdByBugId[current]

        return current

class NonNegativeRandomGenerator(RandomGenerator):
    """
    Generate the negative pairs which the nn loss is bigger than alpha.
    """

    def __init__(self, preprocessor, collate, rate, bugIdList, masterIdByBugId, nTries, device, randomAnchor=True, silence=False, decimals=3):
        """
        """
        super(NonNegativeRandomGenerator, self).__init__(preprocessor, collate, rate, bugIdList, masterIdByBugId, randomAnchor)
        self.nTries = nTries
        self.silence = silence
        self.decimals = decimals
        self.device = device

    def generatePairs(self, model, loss, posPairs, pairsWithId):
        negativePairs = []

        self.setPossibleAnchors(pairsWithId)

        for i in range(self.nTries):
            if not self.silence:
                self.logger.info("==> Try {}".format(i + 1))

            for _ in range(self.rate):
                anchors = self.sampleAnchor(len(posPairs))

                for negPair in self.generateExamples(model, loss, anchors):
                    negativePairs.append(negPair)

                    if len(negativePairs) == len(posPairs) * self.rate:
                        return posPairs, negativePairs

            if not self.silence:
                self.logger.info("==> Try {} - we still have to generate {} good pairs.".format(
                    i + 1, len(posPairs) * self.rate - len(negativePairs)))

        missing_negative = len(posPairs) * self.rate - len(negativePairs)

        if missing_negative > 0:
            self.logger.info(
                "We generated a number of negative pairs (%d) that was insufficient to maintain the same rate." % (
                    len(negativePairs)))
            nPosPairs = int(len(negativePairs) / float(self.rate))
            self.logger.info("Randomly select %d positive pairs to maintain the rate." % (nPosPairs))
            posPairs = random.sample(posPairs, nPosPairs)

        return posPairs, negativePairs

    def generateTriplets(self, model, tripletLoss, posPairs):
        """
        Given a positive pair, the class generates the new examples.
        :param model:
        :param loss:
        :param posPairs:
        :return:
        """
        triplets = []

        for i in range(self.nTries):
            self.logger.info("==> Try {}".format(i + 1))
            for _ in range(self.rate):
                for newTriplet in self.generateExamples(model, tripletLoss, posPairs):
                    triplets.append(newTriplet)

                    if len(triplets) == len(posPairs) * self.rate:
                        return triplets

            self.logger.info("==> Try {} - we still have to generate {} good pairs.".format(
                i + 1, len(posPairs) * self.rate - len(triplets)))

        if len(triplets) < len(posPairs) * self.rate:
            self.logger.info(
                "We generated a number of new triplets (%d) that was insufficient to maintain the same rate." % (
                    len(triplets)))

        return triplets

    def generateExamples(self, model, lossFun, anchors):
        batchSize = 128
        negatives = []
        nIteration = math.ceil(len(anchors) / float(batchSize))
        decimals = self.decimals

        model.eval()

        with torch.no_grad():
            for it in range(nIteration):
                batch = []
                bugIds = []

                for anchor in anchors[it * batchSize: (it + 1) * batchSize]:
                    if isinstance(anchor, (list, tuple)):
                        anchorId, anchorEmb = anchor[0]
                        posId, posEmb = anchor[1]

                        negId = self.sampleNegativeExample(anchorId)
                        negEmb = self.preprocessor.extract(negId)

                        bugIds.append((anchorId, posId, negId))
                        batch.append((anchorEmb, posEmb, negEmb))
                    else:
                        anchorId = anchor
                        anchorEmb = self.preprocessor.extract(anchorId)
                        negId = self.sampleNegativeExample(anchorId)
                        negEmb = self.preprocessor.extract(negId)

                        bugIds.append((anchorId, negId))
                        batch.append((anchorEmb, negEmb, 0.0))

                x = self.collate.collate(batch)
                input, target = self.collate.to(x, self.device)
                output = model(*input)

                lossValue = lossFun(output, target).flatten()
                lossValues, idxs = torch.sort(lossValue, descending=True)

                lossValues = lossValues.data.cpu().numpy().flatten()
                idxs = idxs.data.cpu().numpy().flatten()

                for idx, lossValue in zip(idxs, lossValues):
                    if np.around(lossValue, decimals=decimals) > 0.0:
                        yield batch[idx]