# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb 
from overrides import overrides

from allennlp.training.metrics import Metric
from dataflow.leaderboard.evaluate import evaluate_prediction_exact_match 
from dataflow.core.turn_prediction import TurnPrediction, TurnAnswer
from dataflow.core.dialogue import TurnId, ProgramExecutionOracle

class BasicExactMatch(Metric):
    """
    Accumulator for exact match statistics.
    """

    def __init__(self):
        self.hits = 0
        self.total = 0

    def __call__(self, 
                 true_str: str,
                 pred_str: str,
                 input_str: str = None):

        match=False
        if true_str.strip() == pred_str.strip():
            self.hits += 1
            match=True
        else:
            pass
            #if input_str is not None:
            #    print(f"Input: {input_str}")
        self.total += 1
        return match 

    def get_metric(self, reset: bool = False):
        if self.total == 0:
            return -1
        to_ret = self.hits / self.total
        if reset:
            self.hits = 0
            self.total = 0
        return to_ret 

class AdvancedExactMatch(Metric):
    def __init__(self):
        self.exact_score = 0
        self.no_refer_score = 0
        self.total = 0

    def __call__(self, 
                 true_str: str,
                 pred_str: str,
                 input_str: str = None):
        pred = TurnPrediction(TurnId("test", 0), input_str, pred_str)
        true = TurnAnswer(TurnId("test", 0), input_str, true_str, ProgramExecutionOracle(False, True))

        match, match_no_refer = evaluate_prediction_exact_match(pred, true)
        if not match and input_str is not None:
            print(f"Input: {input_str}")

        if match: 
            self.exact_score += 1
        if match_no_refer:
            self.no_refer_score += 1
        self.total += 1

        return match

    def get_metric(self, reset: bool = False):
        if self.total == 0:
            return -1
        avg_exact_score = self.exact_score / self.total
        avg_no_refer_score = self.no_refer_score / self.total
        if reset:
            self.exact_score = 0
            self.no_refer_score = 0
            self.total = 0
        return avg_exact_score
