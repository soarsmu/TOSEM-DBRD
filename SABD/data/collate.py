import torch

class PairBugCollate(object):

    def __init__(self, inputHandlers, labelDataType=None, ignore_target=False, unsqueeze_target=False):
        self.inputHandlers = inputHandlers
        self.labelDataType = labelDataType
        self.ignore_target = ignore_target
        self.unsqueeze_target = unsqueeze_target

    def to(self, batch, device):
        # num_workers > 0: tensors have the be transfer to the GPU in the main thread.
        x, target = batch
        new_x = []

        for bug_data in x:
            device_module_inputs = []
            for encoder_input in bug_data:
                device_nn_input = []
                for data in encoder_input:
                    if data is None:
                        device_nn_input.append(None)
                    else:
                        device_nn_input.append(data.to(device))
                device_module_inputs.append(device_nn_input)

            new_x.append(device_module_inputs)

        if self.unsqueeze_target:
            target = target.unsqueeze(1)

        if not self.ignore_target:
            target = target.to(device)

        return new_x, target

    def collate(self, batch):
        """
        Prepare data before doing forward propagation and backpropagation.
        Dataset returns bug1 and bug2 and the pair label.
        bug1 and bug2 are lists which each dimension is related to a different information source of the bug.

        :param batch: batch of examples
        :return:
        """
        # For each information type, we will have a specific InputHandler
        bug1InfoBatches = [[] for _ in self.inputHandlers]
        bug2InfoBatches = [[] for _ in self.inputHandlers]
        labels = []

        # Separating X and Y
        for bug1, bug2, label in batch:
            # We put each information type in a same matrix. This matrix is the batch of a specific enconder.
            for infoIdx, infoInput in enumerate(bug1):
                bug1InfoBatches[infoIdx].append(infoInput)

            for infoIdx, infoInput in enumerate(bug2):
                bug2InfoBatches[infoIdx].append(infoInput)

            labels.append(label)

        # Prepare the input to be send to a encoder
        query = [inputHandler.prepare(infoBatch) 
        for inputHandler, infoBatch in zip(self.inputHandlers, bug1InfoBatches)]
        candidate = [inputHandler.prepare(infoBatch) 
        for inputHandler, infoBatch in zip(self.inputHandlers, bug2InfoBatches)]

        # Transform labels to a tensor
        if self.ignore_target:
            target = None
        else:
            target = torch.tensor(labels, dtype=self.labelDataType)

        return (query, candidate), target


class TripletBugCollate(object):

    def __init__(self, inputHandlers):
        self.inputHandlers = inputHandlers

    def to(self, batch, device):
        # num_workers > 0: tensors have the be transfer to the GPU in the main thread.
        x, _ = batch
        new_x = []

        for bug_data in x:
            device_module_inputs = []
            for encoder_input in bug_data:
                device_nn_input = []
                for data in encoder_input:
                    if data is None:
                        device_nn_input.append(None)
                    else:
                        device_nn_input.append(data.to(device))
                device_module_inputs.append(device_nn_input)

            new_x.append(device_module_inputs)

        return new_x, None

    def collate(self, batch):
        """
        Prepare data before doing forward propagation and backpropagation.
        Dataset returns bug1 and bug2 and the pair label.
        bug1 and bug2 are lists which each dimension is related to a different information source of the bug.

        :param batch: batch of examples
        :return:
        """
        # For each information type, we will have a specific InputHandler
        anchorInfoBatches = [[] for _ in self.inputHandlers]
        posInfoBatches = [[] for _ in self.inputHandlers]
        negInfoBatches = [[] for _ in self.inputHandlers]
        labels = []

        # Separating X and Y
        for anchor, pos, neg in batch:
            # We put each information type in a same matrix. This matrix is the batch of a specific enconder.
            for infoIdx, infoInput in enumerate(anchor):
                anchorInfoBatches[infoIdx].append(infoInput)

            for infoIdx, infoInput in enumerate(pos):
                posInfoBatches[infoIdx].append(infoInput)

            for infoIdx, infoInput in enumerate(neg):
                negInfoBatches[infoIdx].append(infoInput)

        # Prepare the input to be send to a encoder
        anchorInput = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in zip(self.inputHandlers, anchorInfoBatches)]
        
        posInput = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in zip(self.inputHandlers, posInfoBatches)]

        negInput = [inputHandler.prepare(infoBatch) for inputHandler, infoBatch in zip(self.inputHandlers, negInfoBatches)]

        return (anchorInput, posInput, negInput), None
