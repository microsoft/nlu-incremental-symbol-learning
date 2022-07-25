# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sequence-to-sequence metrics"""
from typing import Dict
import math
import pdb 

from overrides import overrides
import torch
from allennlp.training.metrics import Metric

import logging


@Metric.register("extended_pointer_generator")
class ExtendedPointerGeneratorMetrics(Metric):

    def __init__(self,
                 loss: float = 0.0,
                 correct_generation_count: int = 0,
                 generation_count: int = 0,
                 correct_source_copy_count: int = 0,
                 source_copy_count: int = 0,
                 correct_target_copy_count: int = 0,
                 target_copy_count: int = 0,
                 correct_hybrid_count: int = 0,
                 hybrid_count: int = 0,
                 interest_loss: float = 0.0,
                 non_interest_loss: float = 0.0) -> None:
        self._loss = loss
        self._interest_loss = interest_loss
        self._non_interest_loss = non_interest_loss
        self._correct_generation_count = correct_generation_count
        self._generation_count = generation_count
        self._correct_source_copy_count = correct_source_copy_count
        self._source_copy_count = source_copy_count
        self._correct_target_copy_count = correct_target_copy_count
        self._target_copy_count = target_copy_count
        self._correct_hybrid_count = correct_hybrid_count
        self._hybrid_count = hybrid_count
        self._total = 0

    @overrides
    def __call__(self,
                 loss: torch.Tensor,
                 prediction: torch.Tensor,
                 generation_outputs: torch.Tensor,
                 valid_generation_mask: torch.Tensor,
                 source_copy_indices: torch.Tensor,
                 valid_source_copy_mask: torch.Tensor,
                 target_copy_indices: torch.Tensor,
                 valid_target_copy_mask: torch.Tensor,
                 interest_loss: torch.Tensor = None,
                 non_interest_loss: torch.Tensor = None) -> None:
        loss, prediction, generation_outputs, valid_generation_mask, \
            source_copy_indices, valid_source_copy_mask, \
            target_copy_indices, valid_target_copy_mask = self.unwrap_to_tensors(
                loss, prediction, generation_outputs, valid_generation_mask,
                source_copy_indices, valid_source_copy_mask, target_copy_indices, valid_target_copy_mask
        )
        # Generation.
        correct_generation_count = (generation_outputs.eq(prediction) & valid_generation_mask).sum().item()
        generation_count = valid_generation_mask.sum().item()
        # Source-side copy.

        correct_source_copy_count = (source_copy_indices.eq(prediction) & valid_source_copy_mask).sum().item()
        source_copy_count = valid_source_copy_mask.sum().item()
        # Target-side copy.
        correct_target_copy_count = (target_copy_indices.eq(prediction) & valid_target_copy_mask).sum().item()
        target_copy_count = valid_target_copy_mask.sum().item()
        # Update numbers.
        self._loss += loss.sum().item()
        if interest_loss is not None:
            # do not accumulate this across batches 
            self._interest_loss += interest_loss.sum().item()
            self._non_interest_loss += non_interest_loss.sum().item()
            self._total += 1
        self._correct_generation_count += correct_generation_count
        self._generation_count += generation_count
        self._correct_source_copy_count += correct_source_copy_count
        self._source_copy_count += source_copy_count
        self._correct_target_copy_count += correct_target_copy_count
        self._target_copy_count += target_copy_count
        self._correct_hybrid_count += correct_generation_count + correct_source_copy_count + correct_target_copy_count
        self._hybrid_count += generation_count + source_copy_count + target_copy_count

    @property
    def xent(self) -> float:
        """ compute cross entropy """
        return self._loss / self._hybrid_count

    @property
    def ppl(self) -> float:
        """ compute perplexity """
        if self._hybrid_count == 0:
            return -1
        return math.exp(min(self._loss / self._hybrid_count, 100))

    def get_metric(self, reset: bool = False) -> Dict:
        if self._total == 0:
            self._interest_loss = 0
            self._non_interest_loss = 0
            self._total = 1
        metrics = {
            #"loss": self._loss, 
            "interest_loss": self._interest_loss / self._total,
            "non_interest_loss": self._non_interest_loss / self._total,
            "accuracy": accuracy(self._correct_hybrid_count, self._hybrid_count),
            "generate": accuracy(self._correct_generation_count, self._generation_count),
            "src_copy": accuracy(self._correct_source_copy_count, self._source_copy_count),
            "tgt_copy": accuracy(self._correct_target_copy_count, self._target_copy_count),
            "gen_freq": accuracy(self._generation_count, self._hybrid_count),
            "src_freq": accuracy(self._source_copy_count, self._hybrid_count),
            "tgt_freq": accuracy(self._target_copy_count, self._hybrid_count),
            "ppl": self.ppl
        }
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self) -> None:
        self._loss = 0.0
        self._interest_loss = 0.0
        self._non_interest_loss = 0.0
        self._correct_generation_count = 0
        self._generation_count = 0
        self._correct_source_copy_count = 0
        self._source_copy_count = 0
        self._correct_target_copy_count = 0
        self._target_copy_count = 0
        self._correct_hybrid_count = 0
        self._hybrid_count = 0
        self._total = 0

def accuracy(correct_count: float, total_count: float) -> float:
    """ compute accuracy """
    if total_count == 0:
        return 0.0
    return correct_count / total_count

@Metric.register("pointer_generator")
class PointerGeneratorMetrics(Metric):

    def __init__(self,
                 loss: float = 0.0,
                 correct_generation_count: int = 0,
                 generation_count: int = 0,
                 correct_source_copy_count: int = 0,
                 source_copy_count: int = 0,
                 correct_hybrid_count: int = 0,
                 hybrid_count: int = 0,
                 interest_loss: float = 0.0,
                 non_interest_loss: float = 0.0) -> None:
        self._loss = loss
        self._interest_loss = interest_loss
        self._non_interest_loss = non_interest_loss
        self._correct_generation_count = correct_generation_count
        self._generation_count = generation_count
        self._correct_source_copy_count = correct_source_copy_count
        self._source_copy_count = source_copy_count
        self._correct_hybrid_count = correct_hybrid_count
        self._hybrid_count = hybrid_count
        self._total = 0

    @overrides
    def __call__(self,
                 loss: torch.Tensor,
                 prediction: torch.Tensor,
                 generation_outputs: torch.Tensor,
                 valid_generation_mask: torch.Tensor,
                 source_copy_indices: torch.Tensor,
                 valid_source_copy_mask: torch.Tensor,
                 interest_loss: torch.Tensor = None,
                 non_interest_loss: torch.Tensor = None,
                 ) -> None:
        loss, prediction, generation_outputs, valid_generation_mask, \
            source_copy_indices, valid_source_copy_mask = self.unwrap_to_tensors(
                loss, prediction, generation_outputs, valid_generation_mask,
                source_copy_indices, valid_source_copy_mask)
        # Generation.
        correct_generation_count = (generation_outputs.eq(prediction) & valid_generation_mask).sum().item()
        generation_count = valid_generation_mask.sum().item()
        # Source-side copy.

        correct_source_copy_count = (source_copy_indices.eq(prediction) & valid_source_copy_mask).sum().item()
        source_copy_count = valid_source_copy_mask.sum().item()
        # Update numbers.
        self._loss += loss.sum().item()
        if interest_loss is not None:
            try:
                self._interest_loss += interest_loss.sum().item()
                self._non_interest_loss += non_interest_loss.sum().item()
            except AttributeError:
                self._interest_loss = 0.0
                self._non_interest_loss = 0.0
            self._total += 1
        self._correct_generation_count += correct_generation_count
        self._generation_count += generation_count
        self._correct_source_copy_count += correct_source_copy_count
        self._source_copy_count += source_copy_count
        self._correct_hybrid_count += correct_generation_count + correct_source_copy_count 
        self._hybrid_count += generation_count + source_copy_count 

    @property
    def xent(self) -> float:
        """ compute cross entropy """
        return self._loss / self._hybrid_count

    @property
    def ppl(self) -> float:
        """ compute perplexity """
        if self._hybrid_count == 0:
            return -1
        return math.exp(min(self._loss / self._hybrid_count, 100))

    def get_metric(self, reset: bool = False) -> Dict:
        if self._total == 0:
            self._interest_loss = 0
            self._non_interest_loss = 0
            self._total = 1
        metrics = {
            "loss": self._loss, 
            "interest_loss": self._interest_loss / self._total,
            "non_interest_loss": self._non_interest_loss / self._total,
            "accuracy": accuracy(self._correct_hybrid_count, self._hybrid_count),
            "generate": accuracy(self._correct_generation_count, self._generation_count),
            "src_copy": accuracy(self._correct_source_copy_count, self._source_copy_count),
            "gen_freq": accuracy(self._generation_count, self._hybrid_count),
            "src_freq": accuracy(self._source_copy_count, self._hybrid_count),
            "ppl": self.ppl
        }
        if reset:
            self.reset()
        return metrics

    @overrides
    def reset(self) -> None:
        self._loss = 0.0
        self._interest_loss = 0.0
        self._non_interest_loss = 0.0
        self._correct_generation_count = 0
        self._generation_count = 0
        self._correct_source_copy_count = 0
        self._source_copy_count = 0
        self._correct_hybrid_count = 0
        self._hybrid_count = 0
        self._total = 0

def accuracy(correct_count: float, total_count: float) -> float:
    """ compute accuracy """
    if total_count == 0:
        return 0.0
    return correct_count / total_count
