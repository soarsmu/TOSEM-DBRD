import logging
from time import time

import math
import torch
from torch.autograd import Variable

from metrics.metric import Accuracy


class SaveModelHandler(object):
    """
    Save the model when it achieves performances better than the preciously ones.
    """

    def __init__(self, metricObj, inverse=False):
        """
        :param metricObj: metric object
        :param inverse: If it is true, the class save the model when the current performance is smaller the best.
        Otherwise
        """
        self.metricObj = metricObj
        self.bestPerformance = math.inf if inverse else -math.inf
        self.inverse = inverse
        self.logger = logging.getLogger(__name__)

    def isToSaveModel(self, validationMetrics):
        for metric in validationMetrics:
            if self.metricObj is metric:
                cur = self.metricObj.lastResult()

                if self.inverse:
                    if cur < self.bestPerformance:
                        self.bestPerformance = cur
                        return True
                else:
                    if cur > self.bestPerformance:
                        self.bestPerformance = cur
                        return True

                return False

        self.logger.warning("The metric was not found in the list. Returning False")
        return False



class ModelHelper:
    """

    """
    def __init__(self, model, loss, useCuda, convertDataToVariable=True, saveAllEpochs=False):
        self.loss = loss
        self.model = model
        self.useCuda = useCuda
        self.convertDataToVariable = convertDataToVariable
        self.logger = logging.getLogger()
        self.saveAllEpochs = saveAllEpochs

    def getVariable(self, batch):
        variables = []

        for d in batch:
            variables.append(Variable(d).cuda() if self.useCuda else Variable(d))

        return variables

    def __resetMetrics(self):
        for metric in self.trainingMetrics():
            metric.reset()

    def __updatedMetrics(self, metrics, loss, output, target):
        for metric in metrics:
            metric.update(loss, output, target)

    def train(self, train_loader, optimizer, trainingMetrics, epoch):
        self.model.train()
        trainingTime = 0

        for batch_idx, batch in enumerate(train_loader):
            beginForward = time()

            if self.convertDataToVariable:
                batch = self.getVariable(batch)
            data = batch[:-1]
            target = batch[-1]

            optimizer.zero_grad()

            # Training Step
            output = self.model(*data)  # calls the forward function
            loss = self.loss(output, target)
            loss.backward()
            optimizer.step()

            # Calculate training time
            trainingTime += time() - beginForward

            # Update the metrics with new values
            self.__updatedMetrics(trainingMetrics, loss, output, target)

        # Printing training time and metric values
        self.logger.info(" Training Time: %.2f" % trainingTime)

        for metric in trainingMetrics:
            metric.logResult(epoch,'training')


        return trainingMetrics

    def valid(self, valid_loader, validationMetrics, epoch=None):
        self.model.eval()

        for batch in valid_loader:
            if self.convertDataToVariable:
                batch = self.getVariable(batch)
            data = batch[:-1]
            target = batch[-1]

            output = self.model(*data)

            # Update the metrics with new values
            self.__updatedMetrics(validationMetrics, None, output, target)

        # Printing metric values
        self.logger.info('Validation')

        for metric in validationMetrics:
            metric.logResult(epoch, 'validation')

        self.logger.info('')

        return validationMetrics

    def experiment(self, epochs, optimizer, train_loader, valid_loader, trainingMetrics, validationMetrics, decisionToSaveModel=None, fileToSaveModel=None, parametersToSave={}):
        best_acc = 0

        if fileToSaveModel and decisionToSaveModel is None:
            # We use accuracy as the default metric to save the model
            metricObj = None
            for metric in validationMetrics:
                if isinstance(metricObj, Accuracy):
                    metricObj = metric
                    break

            if metricObj:
                self.logger.info("Using accuracy to compare performance")
                decisionToSaveModel = SaveModelHandler(metricObj)

        if self.useCuda:
            self.model.cuda()

        if not isinstance(parametersToSave, dict):
            self.logger.error("parametersToSave have to be a dictionary")
            exit(-1)

        for epoch in range(1, epochs + 1):
            self.logger.info("Epoch: %d" % epoch)
            self.train(train_loader, optimizer, trainingMetrics, epoch)
            self.valid(valid_loader, validationMetrics, epoch)

            if fileToSaveModel:
                if self.saveAllEpochs or (decisionToSaveModel and decisionToSaveModel.isToSaveModel(validationMetrics)):
                    parametersToSave['model'] = self.model.state_dict()
                    self.logger.info("==> Saving Model: %s" % fileToSaveModel)
                    torch.save(parametersToSave, fileToSaveModel)

        return self.model, best_acc, trainingMetrics, validationMetrics
