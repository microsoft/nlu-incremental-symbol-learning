# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides
import torch
from allennlp.common.registrable import Registrable
import logging
import pdb 
logger = logging.getLogger(__name__)

class LabelSmoothing(torch.nn.Module, Registrable):
    """Implement label smoothing."""
    def __init__(self,
                 pad_index: int = 0,
                 smoothing: float = 0.0) -> None:
        super().__init__()
        self._criterion = torch.nn.KLDivLoss(reduction="sum")
        self._pad_index = pad_index
        self._smoothing = smoothing
        self._confidence = 1.0 - smoothing

    def reset_parameters(self,
                         pad_index: int = None,
                         smoothing: float = None) -> None:
        if pad_index is not None:
            self._pad_index = pad_index
        if smoothing is not None:
            self._smoothing = smoothing
            self._confidence = 1.0 - smoothing

    @overrides
    def forward(self,
                x: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        :param x: log-probs [num_instances, vocab_size]
        :param target: [num_instances]
        """
        vocab_size = x.size(1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self._smoothing / (vocab_size - 2))  # Exclude pad and target.
        true_dist.scatter_(1, target.unsqueeze(1), self._confidence)
        true_dist[:, self._pad_index] = 0
        mask = target.eq(self._pad_index)
        true_dist.masked_fill_(mask.unsqueeze(1), 0.0)

        return self._criterion(x, true_dist)

@LabelSmoothing.register("base")
class BaseLabelSmoothing(LabelSmoothing):
    def __init__(self,
                 pad_index: int = 0,
                 smoothing: float = 0.0) -> None:
        super().__init__(pad_index=pad_index,
                         smoothing=smoothing)


@LabelSmoothing.register("no_sum")
class NoSumLabelSmoothing(LabelSmoothing):
    """Implement label smoothing."""

    def __init__(self,
                 pad_index: int = 0,
                 smoothing: float = 0.0) -> None:
        super().__init__(pad_index=pad_index, smoothing=smoothing)

        # remove sum reduction from loss so that we can reweight it 
        self._criterion = torch.nn.KLDivLoss(reduction='none')

