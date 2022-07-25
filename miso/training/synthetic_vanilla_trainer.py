# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import numpy
import subprocess
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any
import sys
from overrides import overrides
import time
import datetime
import traceback
import math
import os
import re
import pdb 

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  lazy_groups_of)
from allennlp.training import Trainer
from allennlp.training.trainer_pieces import TrainerPieces
from allennlp.training.trainer_base import TrainerBase
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models import Model
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer

from miso.data.dataset_readers.calflow_parsing.calflow_sequence import CalFlowSequence
from miso.metrics.exact_match import BasicExactMatch, AdvancedExactMatch
from miso.metrics.fxn_metrics import SingleFunctionMetric, SyntheticFunctionMetric
from miso.training.calflow_trainer import CalflowTrainer
from miso.training.calflow_vanilla_trainer import VanillaCalflowTrainer
#from miso.data.iterators.data_iterator import DecompDataIterator, DecompBasicDataIterator 
from dataflow.core.lispress import render_compact, parse_lispress

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("vanilla_synthetic_parsing")
class SyntheticVanillaCalflowTrainer(VanillaCalflowTrainer):

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exact_match_metric = BasicExactMatch()
        self.fxn_metric = SyntheticFunctionMetric(self.model.fxn_of_interest)

    @overrides
    def _update_validation_exact_match(self, pred_instances: List[Dict[str, numpy.ndarray]],
                                             true_instances):

        logger.info("Computing Exact Match")

        for batch in true_instances:
            assert(len(batch) == 1)

        true_sents = [" ".join(true_inst) for batch in true_instances for true_inst in batch[0]['src_tokens_str']]
        # remove start and end tokens 
        true_graphs = [true_inst for batch in true_instances for true_inst in batch[0]['tgt_tokens_inputs']]

        pred_graphs = [" ".join(pred_inst['nodes']) 
                         for i, pred_inst in enumerate(pred_instances)]

        true_lispress_strs = [ts for ts in true_graphs]
        pred_lispress_strs = [ps for ps in pred_graphs]

        for ts, ps, inp_str in zip(true_lispress_strs, pred_lispress_strs, true_sents):
            self.exact_match_metric(ts, ps, inp_str) 
            if self.model.fxn_of_interest is not None:
                self.fxn_metric(ts, ps) 

        exact_match_score = self.exact_match_metric.get_metric(reset=True)
        self.model.exact_match_score = exact_match_score

        if self.model.fxn_of_interest is not None:
            coarse, fine, __, __, __ = self.fxn_metric.get_metric(reset = True)
            self.model.coarse_fxn_metric = coarse
            self.model.fine_fxn_metric = fine 

