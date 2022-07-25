# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from miso.metrics.exact_match import AdvancedExactMatch, BasicExactMatch
import re
import pdb 
from allennlp.training.metrics import Metric
from dataflow.core.lispress import parse_lispress, render_compact
from miso.metrics.exact_match import AdvancedExactMatch
import logging 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class SingleFunctionMetric(Metric):
    def __init__(self, fxn_name: str):
        self.fxn_name = fxn_name 
        self.coarse_grained_score = 0
        self.false_positives = 0
        self.true_positives = 0
        self.false_negatives = 0
        self.fine_grained_score = 0
        self.total = 0
        self.pred_total = 0
        self.exact_match_call = AdvancedExactMatch()

    def __call__(self, 
                 true_str: str,
                 pred_str: str):
        # normalize
        try:
            true_str = render_compact(parse_lispress(true_str))
        except:
            logger.info(f"skipping string {true_str} because it's malformed")
            true_str = "(Skip)"
        
        pred_str = render_compact(parse_lispress(pred_str))
        
        split_true = [x.strip() if x is not None else "" for x in re.split("[() ]", true_str)]
        split_pred = [x.strip() if x is not None else "" for x in re.split("[() ]", pred_str)]

        if self.fxn_name in split_true:
            self.total += 1
            if self.exact_match_call(true_str, pred_str):
                self.true_positives += 1
                self.fine_grained_score += 1
                self.coarse_grained_score += 1
            else:
                if self.fxn_name in split_pred:
                    self.coarse_grained_score += 1
                    self.true_positives += 1

        if self.fxn_name in split_true and self.fxn_name not in split_pred:
            self.false_negatives += 1
        if self.fxn_name not in split_true and self.fxn_name in split_pred:
            self.false_positives += 1

    def get_metric(self, reset: bool = False):
        if self.total == 0:
            return -1, -1, -1, -1, -1
        coarse = self.coarse_grained_score/self.total
        fine = self.fine_grained_score/self.total
        try:
            precision = self.true_positives/(self.true_positives + self.false_positives)
        except ZeroDivisionError:
            precision = -1
        try:
            recall = self.true_positives/(self.true_positives + self.false_negatives)
        except ZeroDivisionError:
            recall = -1
        try:
            f1 = (2 * precision * recall)/(precision + recall)
        except ZeroDivisionError:
            f1 = -1
        to_ret = (coarse, fine, precision, recall, f1)

        if reset:
            self.coarse_grained_score = 0
            self.fine_grained_score = 0
            self.total = 0
            self.true_positives = 0
            self.false_positives = 0
            self.false_negatives = 0

        return to_ret 

class SyntheticFunctionMetric(SingleFunctionMetric):
    def __init__(self, fxn_name: str):
        super().__init__(fxn_name) 
        self.exact_match_call = BasicExactMatch()

    def __call__(self, 
                 true_str: str,
                 pred_str: str):

        true_str = true_str.strip()
        pred_str = pred_str.strip()  
        split_true = [x.strip() if x is not None else "" for x in re.split(" ", true_str)]
        split_pred = [x.strip() if x is not None else "" for x in re.split(" ", pred_str)]

        if self.fxn_name in split_true:
            self.total += 1
            if self.exact_match_call(true_str, pred_str):
                self.true_positives += 1
                self.fine_grained_score += 1
                self.coarse_grained_score += 1
            else:
                if self.fxn_name in split_pred:
                    self.coarse_grained_score += 1
                    self.true_positives += 1

        if self.fxn_name in split_true and self.fxn_name not in split_pred:
            self.false_negatives += 1
        if self.fxn_name not in split_true and self.fxn_name in split_pred:
            self.false_positives += 1

    def get_metric(self, reset: bool = False):
        if self.total == 0:
            return -1, -1, -1, -1, -1
        coarse = self.coarse_grained_score/self.total
        fine = self.fine_grained_score/self.total
        try:
            precision = self.true_positives/(self.true_positives + self.false_positives)
        except ZeroDivisionError:
            precision = -1
        try:
            recall = self.true_positives/(self.true_positives + self.false_negatives)
        except ZeroDivisionError:
            recall = -1
        try:
            f1 = (2 * precision * recall)/(precision + recall)
        except ZeroDivisionError:
            f1 = -1
        to_ret = (coarse, fine, precision, recall, f1)

        if reset:
            self.coarse_grained_score = 0
            self.fine_grained_score = 0
            self.total = 0
            self.true_positives = 0
            self.false_positives = 0
            self.false_negatives = 0

        return to_ret 
