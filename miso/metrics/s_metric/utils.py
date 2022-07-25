# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-


import sys
import logging
import random
import logging


logger = logging.getLogger(__name__)

def set_seed(seed):
    """Sets random seed everywhere."""
    random.seed(seed)


def get_logging():
    log = logger
    if log.handlers:
        return log
    # formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    # ch.setFormatter(formatter)
    return log


def wc(files):
    if type(files) is list or type(files) is tuple:
        pass
    else:
        files = [files]
    return sum([sum([1 for _ in open(file)]) for file in files])


def compute_f(match_num, test_num, gold_num):
    """
    Compute the f-score based on the matching triple number,
                                 triple number of AMR set 1,
                                 triple number of AMR set 2
    Args:
        match_num: matching triple number
        test_num:  triple number of AMR 1 (test file)
        gold_num:  triple number of AMR 2 (gold file)
    Returns:
        precision: match_num/test_num
        recall: match_num/gold_num
        f_score: 2*precision*recall/(precision+recall)
    """
    if test_num == 0 or gold_num == 0:
        return 0.00, 0.00, 0.00
    precision = float(match_num) / float(test_num)
    recall = float(match_num) / float(gold_num)
    if (precision + recall) != 0:
        f_score = 2 * precision * recall / (precision + recall)
        return precision, recall, f_score
    else:
        return precision, recall, 0.00
