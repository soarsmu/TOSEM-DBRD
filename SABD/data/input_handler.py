import numpy as np
import torch

from util.torch_util import padSequences


class BasicInputHandler(object):

    def __init__(self, transpose_input=True):
        self.transpose_input = transpose_input

    def prepare(self, batch):
        """
        This function is call when one batch is built. We transform the batch input in a torch variable.
        :param batch:
        :return:
        """
        if isinstance(batch, (list)):
            arr = np.asarray(batch)
        else:
            arr = np.empty(len(batch))

            for i, el in batch:
                arr[i] = el

            arr = arr.t

        if self.transpose_input:
            arr = arr.T

        return (torch.from_numpy(arr),)


class SABDInputHandler(object):

    def __init__(self, padding_idx, min_input_size=-1, field_padding_idx=0):
        self.padding_idx = padding_idx
        self.field_padding_idx = field_padding_idx
        self.min_input_size = min_input_size

    def prepare(self, batch):
        """
        This function is call when one batch is built.
        :param batch:
        :return:
        """
        textual_input = []
        field_input = []
        tf_input = []

        sizes = []
        last_size = -1
        has_same_size = True

        for txt, field, tf in batch:
            textual_input.append(txt)
            field_input.append(field)

            if tf is not None:
                tf_input.append(tf)

            sizes.append(self.min_input_size if len(txt) < self.min_input_size else len(txt))
            if has_same_size:
                if last_size != -1:
                    has_same_size = last_size == sizes[-1]

                last_size = sizes[-1]

        if has_same_size and self.min_input_size <= last_size:
            tf_input = None if len(tf_input) == 0 else torch.tensor(tf_input, dtype=torch.float32)

            return (torch.tensor(textual_input, dtype=torch.int64),
                    torch.tensor(field_input, dtype=torch.int64),
                    tf_input,
                    torch.tensor(sizes))

        tf_input = None if len(tf_input) == 0 else torch.from_numpy(
            padSequences(tf_input, 0.0, dtype='float32', minSize=self.min_input_size))

        return (
            torch.from_numpy(padSequences(textual_input, self.padding_idx, dtype="int64", minSize=self.min_input_size)),
            torch.from_numpy(
                padSequences(field_input, self.field_padding_idx, dtype="int64", minSize=self.min_input_size)),
            tf_input,
            torch.tensor(sizes))


class RNNInputHandler(object):
    LENGTH_IDX = 2

    def __init__(self, padding_idx, min_input_size=-1):
        self.padding_idx = padding_idx
        self.min_input_size = min_input_size

    def prepare(self, batch):
        """
        This function is call when one batch is built.
        :param batch:
        :return:
        """
        return (
            torch.from_numpy(padSequences(batch, self.padding_idx, dtype="int64", minSize=self.min_input_size)),
            None,
            torch.tensor([self.min_input_size if len(seq) < self.min_input_size else len(seq) for seq in batch]))


class TextCNNInputHandler(object):

    def __init__(self, padding_idx, min_input_size=-1):
        self.padding_idx = padding_idx
        # An error occurs when the input size of a convolution is smaller than the kernel size.
        # We pad the input to have at least the same size of biggest kernel size.
        self.min_input_size = min_input_size

    def prepare(self, batch):
        """
        This function is call when one batch is built.
        :param batch:
        :return:
        """
        return (torch.from_numpy(padSequences(batch, self.padding_idx, dtype="int64", minSize=self.min_input_size)),)