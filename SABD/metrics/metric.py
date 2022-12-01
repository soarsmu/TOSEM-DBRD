import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Loss, Metric, Accuracy, Precision, Recall
import numpy as np
import numbers

from ignite.metrics import Metric, MetricsLambda
from ignite.exceptions import NotComputableError


class PredictionCache(Metric):

    def __init__(self, output_transform=lambda x: x):
        super(PredictionCache, self).__init__(output_transform)
        self.predictions = []
        self.reset()

    def reset(self):
        self.predictions = []

    def update(self, output):
        y_pred, y = output

        if len(y_pred.size()) == 1:
            indices = torch.round(y_pred).type(y.type())
        else:
            indices = torch.max(y_pred, dim=1)[1]

        self.predictions.extend(map(lambda x: np.float(x), indices))

    def compute(self):
        return 0.0


# class PrecisionRecallF1(object):
#     """
#     Compute precision, recall and F1 over a sequence of mini-batches.
#     """
#
#     def __init__(self, predictionFunc, labels=None, average=None, pos_label=1):
#         # Number of examples seen so far.
#         self.numExamples = 0
#         # Accumulated accuracy over all examples seen so far.
#         self.accumAccuracy = 0.0
#
#         self.labels = labels
#         self.average = average
#         self.pos_label = pos_label
#         self.predictions = []
#         self.targets = []
#         self.results = []
#
#         self.predictionFunc = predictionFunc
#         self.logger = logging.getLogger(__name__)
#
#     def update(self, losses, outputs, targets):
#         predictions = self.predictionFunc(outputs).detach().cpu().numpy().flatten()
#
#         for t, p in zip(targets, predictions):
#             self.predictions.append(p)
#             self.targets.append(t)
#
#     def getResult(self):
#         prec, recall, f1, _ = precision_recall_fscore_support(self.targets, self.predictions, labels=self.labels,
#                                                               average=self.average, pos_label=self.pos_label)
#
#         self.results.append((list(prec), list(recall), list(f1)))
#
#         self.reset()
#
#         return prec, recall, f1
#
#     def lastResult(self):
#         return self.results[-1]
#
#     def reset(self):
#         self.predictions = []
#         self.targets = []
#
#     def logResult(self, epoch, label=''):
#         prec, recall, f1 = self.getResult()
#
#         self.logger.info({'type': 'metric', 'label': '%s_precision' % (label), 'value': list(prec), 'epoch': epoch})
#         self.logger.info({'type': 'metric', 'label': '%s_recall' % (label), 'value': list(recall), 'epoch': epoch})
#         self.logger.info({'type': 'metric', 'label': '%s_f1' % (label), 'value': list(f1), 'epoch': epoch})
#
#
# class PrecisionRecallF1Triplets(PrecisionRecallF1):
#
#     def update(self, losses, outputs, targets):
#         bugEmbeddings, duplicateEmbeddings, nonDuplicateEmbeddings = outputs
#         batchSize = outputs[0].shape[0]
#
#         outs = [bugEmbeddings, duplicateEmbeddings]
#         targets = torch.ones(batchSize)
#         super(PrecisionRecallF1Triplets, self).update(losses, outs, targets)
#
#         outs = [bugEmbeddings, nonDuplicateEmbeddings]
#         targets = torch.zeros(batchSize)
#         super(PrecisionRecallF1Triplets, self).update(losses, outs, targets)

class AccuracyWrapper(Accuracy):

    def compute(self):
        return {'value': super(AccuracyWrapper, self).compute(), 'num_correct': self._num_correct, 'num_example': self._num_examples}

class PrecisionWrapper(Precision):

    def compute(self):
        return {'value': super(PrecisionWrapper, self).compute().item(), 'positive': self._true_positives.item(), 'total': self._positives.item()}

class RecallWrapper(Recall):

    def compute(self):
        return {'value': super(RecallWrapper, self).compute().item(), 'positive': self._true_positives.item(), 'total': self._positives.item()}


# class Accuracy(object):
#     """
#     Compute precision accuracy over a sequence of mini-batches.
#     """
#
#     def __init__(self, predictionFunc):
#         # Number of examples seen so far.
#         self.numExamples = 0
#         # Accumulated accuracy over all examples seen so far.
#         self.accumAccuracy = 0.0
#         self.results = []
#         self.predictionFunc = predictionFunc
#         self.logger = logging.getLogger(__name__)
#
#     def update(self, losses, outputs, targets):
#         predictions = self.predictionFunc(outputs).detach().cpu().numpy().flatten()
#         targets = targets.detach().cpu().numpy()
#
#         self.accumAccuracy += np.float(accuracy_score(targets, predictions, normalize=False))
#         self.numExamples += len(targets)
#
#     def getResult(self):
#         accum = self.accumAccuracy
#         nEXamples = self.numExamples
#         acc = self.accumAccuracy / self.numExamples
#
#         self.results.append(acc)
#
#         self.reset()
#
#         return acc, accum, nEXamples
#
#     def lastResult(self):
#         return self.results[-1]
#
#     def reset(self):
#         self.numExamples = 0
#         self.accumAccuracy = 0.0
#
#     def logResult(self, epoch, label=''):
#         acc, accum, examples = self.getResult()
#
#         self.logger.info({'type': 'metric', 'label': '%s_acc' % label, 'value': acc, 'epoch': epoch})


class AccuracyTripletBugs(Accuracy):
    """
    Calculate accuracy of a triple of bugs. We break each triplet in pairs of bug and duplicate bug and bug and non-duplicate bug.
    After that, we calculate the accuracy in the same manner that we compute it using pairs.
    """

    def update(self, losses, outputs, targets):
        bugEmbeddings, duplicateEmbeddings, nonDuplicateEmbeddings = outputs
        batchSize = outputs[0].shape[0]

        outs = [bugEmbeddings, duplicateEmbeddings]
        targets = torch.ones(batchSize)
        super(AccuracyTripletBugs, self).update(losses, outs, targets)

        outs = [bugEmbeddings, nonDuplicateEmbeddings]
        targets = torch.zeros(batchSize)
        super(AccuracyTripletBugs, self).update(losses, outs, targets)


class MeanScoreDistance(Metric):
    def __init__(self, negTarget=0.0, posTarget=1.0, output_transform=lambda x: x, batch_size=lambda x: x.shape[0]):
        super(MeanScoreDistance, self).__init__(output_transform)
        self._batch_size = batch_size
        self.negTarget = negTarget
        self.posTarget = posTarget

        self.reset()

    def reset(self):
        self.sumPositiveDis = 0.0
        self.sumNegativeDis = 0.0
        self.negTotal = 0.0
        self.posTotal = 0.0

    def update(self, output):
        y_pred, y = output

        if y is None:
            posScore, negScore = y_pred
        else:
            posScore = negScore = y_pred

        self.posMask = (y == self.posTarget).float()
        self.negMask = (y == self.negTarget).float()

        self.sumPositiveDis += (torch.abs(self.posTarget - posScore) * self.posMask).sum().item()
        self.sumNegativeDis += (torch.abs(negScore - self.negTarget) * self.negMask).sum().item()

        self.posTotal += self.posMask.sum().item()
        self.negTotal += self.negMask.sum().item()

    def compute(self):
        if self.posTotal == 0 or self.negTotal == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return (self.sumNegativeDis / self.negTotal, self.sumPositiveDis / self.posTotal)


class LossWrapper(Loss):

    def compute(self):
        return {'value': super(LossWrapper, self).compute(), 'sum': self._sum, 'num_example': self._num_examples}


class AverageLoss(Metric):
    """
    This loss receives only the mean loss value and batch size.
    """

    def __init__(self, loss_fn, output_transform=lambda x: x, batch_size=lambda x: x.shape[0]):
        super(AverageLoss, self).__init__(output_transform)
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        average_loss = output[0]

        if isinstance(output[1], (tuple,list)):
            N = self._batch_size(output[1][0])
        else:
            N = self._batch_size(output[1])

        self._sum += average_loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return {'value': self._sum / self._num_examples, 'sum': self._sum, 'num_example': self._num_examples}


class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multi-class data.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.
    Args:
        num_classes (int): number of classes. See notes for more details.
        average (str, optional): confusion matrix values averaging schema: None, "samples", "recall", "precision".
            Default is None. If `average="samples"` then confusion matrix values are normalized by the number of seen
            samples. If `average="recall"` then confusion matrix values are normalized such that diagonal values
            represent class recalls. If `average="precision"` then confusion matrix values are normalized such that
            diagonal values represent class precisions.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    Note:
        In case of the targets `y` in `(batch_size, ...)` format, target indices between 0 and `num_classes` only
        contribute to the confusion matrix and others are neglected. For example, if `num_classes=20` and target index
        equal 255 is encountered, then it is filtered out.
    """

    def __init__(self, num_classes, average=None, output_transform=lambda x: x):
        if average is not None and average not in ("samples", "recall", "precision"):
            raise ValueError("Argument average can None or one of ['samples', 'recall', 'precision']")

        self.num_classes = num_classes
        self._num_examples = 0
        self.average = average
        self.confusion_matrix = None
        super(ConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64, device='cpu')
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y_pred.ndimension() < 2:
            raise ValueError("y_pred must have shape (batch_size, num_categories, ...), "
            "but given {}".format(y_pred.shape))

        if y_pred.shape[1] != self.num_classes:
            raise ValueError("y_pred does not have correct number of categories: {} vs {}".format(y_pred.shape[1], self.num_classes))

        if not (y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y_pred must have shape (batch_size, num_categories, ...) and y must have "
                "shape of (batch_size, ...), "
                "but given {} vs {}.".format(y.shape, y_pred.shape)
            )

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

    def update(self, output):
        self._check_shape(output)
        y_pred, y = output

        self._num_examples += y_pred.shape[0]

        # target is (batch_size, ...)
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        y = y.flatten()

        target_mask = (y >= 0) & (y < self.num_classes)
        y = y[target_mask]
        y_pred = y_pred[target_mask]

        indices = self.num_classes * y + y_pred
        m = torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += m.to(self.confusion_matrix)

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        if self.average:
            self.confusion_matrix = self.confusion_matrix.float()
            if self.average == "samples":
                return self.confusion_matrix / self._num_examples
            elif self.average == "recall":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=1) + 1e-15)
            elif self.average == "precision":
                return self.confusion_matrix / (self.confusion_matrix.sum(dim=0) + 1e-15)
        return self.confusion_matrix


def IoU(cm, ignore_index=None):
    """Calculates Intersection over Union
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        ignore_index (int, optional): index to ignore, e.g. background index
    Returns:
        MetricsLambda
    Examples:
    .. code-block:: python
        train_evaluator = ...
        cm = ConfusionMatrix(num_classes=num_classes)
        IoU(cm, ignore_index=0).attach(train_evaluator, 'IoU')
        state = train_evaluator.run(train_dataset)
        # state.metrics['IoU'] -> tensor of shape (num_classes - 1, )
    """
    if not isinstance(cm, ConfusionMatrix):
        raise TypeError("Argument cm should be instance of ConfusionMatrix, but given {}".format(type(cm)))

    if ignore_index is not None:
        if not (isinstance(ignore_index, numbers.Integral) and 0 <= ignore_index < cm.num_classes):
            raise ValueError("ignore_index should be non-negative integer, but given {}".format(ignore_index))

    # Increase floating point precision
    cm = cm.type(torch.float64)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)
    if ignore_index is not None:

        def ignore_index_fn(iou_vector):
            if ignore_index >= len(iou_vector):
                raise ValueError(
                    "ignore_index {} is larger than the length of IoU vector {}".format(ignore_index, len(iou_vector)))
            indices = list(range(len(iou_vector)))
            indices.remove(ignore_index)
            return iou_vector[indices]

        return MetricsLambda(ignore_index_fn, iou)
    else:
        return iou


def mIoU(cm, ignore_index=None):
    """Calculates mean Intersection over Union
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        ignore_index (int, optional): index to ignore, e.g. background index
    Returns:
        MetricsLambda
    Examples:
    .. code-block:: python
        train_evaluator = ...
        cm = ConfusionMatrix(num_classes=num_classes)
        mIoU(cm, ignore_index=0).attach(train_evaluator, 'mean IoU')
        state = train_evaluator.run(train_dataset)
        # state.metrics['mean IoU'] -> scalar
    """
    return IoU(cm=cm, ignore_index=ignore_index).mean()


def cmAccuracy(cm):
    """
    Calculates accuracy using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
    Returns:
        MetricsLambda
    """
    # Increase floating point precision
    cm = cm.type(torch.float64)
    return cm.diag().sum() / (cm.sum() + 1e-15)


def cmPrecision(cm, average=True):
    """
    Calculates precision using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        average (bool, optional): if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision
    cm = cm.type(torch.float64)
    precision = cm.diag() / (cm.sum(dim=0) + 1e-15)
    if average:
        return precision.mean()
    return precision


def cmRecall(cm, average=True):
    """
    Calculates recall using :class:`~ignite.metrics.ConfusionMatrix` metric.
    Args:
        cm (ConfusionMatrix): instance of confusion matrix metric
        average (bool, optional): if True metric value is averaged over all classes
    Returns:
        MetricsLambda
    """

    # Increase floating point precision
    cm = cm.type(torch.float64)
    recall = cm.diag() / (cm.sum(dim=1) + 1e-15)
    if average:
        return recall.mean()
    return recall
