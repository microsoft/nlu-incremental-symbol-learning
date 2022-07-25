# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Continous-variable metrics"""
import math

from overrides import overrides

from allennlp.training.metrics import Metric


class ContinuousMetric(Metric):
    """
    Accumulator for loss statistics.
    Currently calculates:
    * MSE
    * Pearson's r
    """

    def __init__(self, prefix, mse_val = 0.0):
        self._prefix = prefix
        self._mse_val = mse_val
        self._n = 0

    def __call__(self, 
                mse_val):

        self._mse_val += mse_val 
        self._n += 1

    def mse(self):
        return self._mse_val 

    def get_metric(self, reset: bool = False):
        if self._n != 0:
            metrics = {
                    f"{self._prefix}_MSE":self.mse()/self._n
                }
        else:
            metrics = {
                    f"{self._prefix}_MSE":-0.0
                }
        return metrics


