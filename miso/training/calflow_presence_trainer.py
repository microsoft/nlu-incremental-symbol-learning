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

from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.metrics.exact_match import BasicExactMatch, AdvancedExactMatch
from miso.metrics.fxn_metrics import SingleFunctionMetric
#from miso.data.iterators.data_iterator import DecompDataIterator, DecompBasicDataIterator 
from dataflow.core.lispress import render_compact

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("calflow_presence_parsing")
class CalflowTrainer(Trainer):

    def __init__(self,
                 validation_data_path: str,
                 validation_prediction_path: str,
                 warmup_epochs: int = 0,
                 accumulate_batches: int = 1,
                 bert_optimizer: Optimizer = None,
                 do_train_metrics: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_data_path = validation_data_path
        self.validation_prediction_path = validation_prediction_path
        self.accumulate_batches = accumulate_batches
        self.bert_optimizer = bert_optimizer
        self.do_train_metrics = do_train_metrics

        self._warmup_epochs = warmup_epochs
        self._curr_epoch = 0

        # metrics 
        self.exact_match_metric = AdvancedExactMatch()
        if self.model.fxn_of_interest is not None:
            self.fxn_metric = SingleFunctionMetric(self.model.fxn_of_interest)

    def _update_validation_exact_match(self, pred_instances: List[Dict[str, numpy.ndarray]],
                                             true_instances):

        logger.info("Computing Exact Match")

        for batch in true_instances:
            assert(len(batch) == 1)

        true_graphs = [true_inst for batch in true_instances for true_inst in batch[0]['calflow_graph'] ]
        true_sents = [" ".join(true_inst) for batch in true_instances for true_inst in batch[0]['src_tokens_str']]


        pred_graphs = [CalFlowGraph.from_prediction(true_sents[i],
                                                    pred_inst['nodes'], 
                                                    pred_inst['node_indices'], 
                                                    pred_inst['edge_heads'], 
                                                    pred_inst['edge_types']) for i, pred_inst in enumerate(pred_instances)]

        true_lispress_strs = [render_compact(tg.lispress) for tg in true_graphs]
        pred_lispress_strs = [render_compact(pg.lispress) for pg in pred_graphs]

        for ts, ps, inp_str in zip(true_lispress_strs, pred_lispress_strs, true_sents):
            self.exact_match_metric(ts, ps, inp_str) 
            if self.model.fxn_of_interest is not None:
                self.fxn_metric(ts, ps) 

        exact_match_score = self.exact_match_metric.get_metric(reset=True)
        self.model.exact_match_score = exact_match_score

        if self.model.fxn_of_interest is not None:
            coarse, fine, prec, rec, f1 = self.fxn_metric.get_metric(reset = True)
            self.model.coarse_fxn_metric = coarse
            self.model.fine_fxn_metric = fine 

    def _validation_forward(self, batch_group: List[TensorDict]) \
            -> TensorDict:
        """
        Does a forward pass on the given batches and returns the output dict (key, value)
        where value has the shape: [batch_size, *].
        """
        assert len(batch_group) == 1
        batch = batch_group[0]
        batch = nn_util.move_to_device(batch, self._cuda_devices[0])
        output_dict = self.model(**batch)

        return output_dict


    @overrides
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())


        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        cumulative_batch_size = 0
        loss = 0.0
        train_true_instances = []
        train_outputs: List[Dict[str, numpy.ndarray]] = []
        for batch_group in train_generator_tqdm:

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            #if batches_this_epoch % self.accumulate_batches == 0: 
            self.optimizer.zero_grad()
            if self.bert_optimizer is not None:
                self.bert_optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)
            if torch.isinf(loss):
                logger.warn(f"NaN loss enountered! Skipping batch {batches_this_epoch}")
                continue

            loss.backward()


            # accumulate over number of batches 
            if batches_this_epoch % self.accumulate_batches == 0:
                if torch.isnan(loss):
                    raise ValueError("nan loss encountered")


                train_loss += loss.item()

                batch_grad_norm = self.rescale_gradients()

                # This does nothing if batch_num_total is None or you are using a
                # scheduler which doesn't update per batch.
                if self._learning_rate_scheduler:
                    self._learning_rate_scheduler.step_batch(batch_num_total)
                if self._momentum_scheduler:
                    self._momentum_scheduler.step_batch(batch_num_total)

                if self._tensorboard.should_log_histograms_this_batch():
                    # get the magnitude of parameter updates for logging
                    # We need a copy of current parameters to compute magnitude of updates,
                    # and copy them to CPU so large models won't go OOM on the GPU.
                    param_updates = {name: param.detach().cpu().clone()
                                     for name, param in self.model.named_parameters()}
                    self.optimizer.step()
                    if self.bert_optimizer is not None:
                        self.bert_optimizer.step() 
                    for name, param in self.model.named_parameters():
                        param_updates[name].sub_(param.detach().cpu())
                        update_norm = torch.norm(param_updates[name].view(-1, ))
                        param_norm = torch.norm(param.view(-1, )).cpu()
                        self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                           update_norm / (param_norm + 1e-7))
                else:
                    self.optimizer.step()
                    if self.bert_optimizer is not None:
                        self.bert_optimizer.step() 
                # zero grads after step 
                loss = 0.0 

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size/batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(self.model, train_loss, batches_this_epoch, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None):
        pieces = TrainerPieces.from_params(params,  # pylint: disable=no-member
                                           serialization_dir,
                                           recover,
                                           cache_directory,
                                           cache_prefix)
        return _from_params(cls,
                            pieces.model,
                            serialization_dir,
                            pieces.iterator,
                            pieces.train_dataset,
                            pieces.validation_dataset,
                            pieces.params,
                            pieces.validation_iterator)


# An ugly way to inherit ``from_params`` of the ``Trainer`` class in AllenNLP.
def _from_params(cls,  # type: ignore
                 model: Model,
                 serialization_dir: str,
                 iterator: DataIterator,
                 train_data: Iterable[Instance],
                 validation_data: Optional[Iterable[Instance]],
                 params: Params,
                 validation_iterator: DataIterator = None) -> CalflowTrainer:
    # pylint: disable=arguments-differ
    patience = params.pop_int("patience", None)
    validation_metric = params.pop("validation_metric", "-loss")
    shuffle = params.pop_bool("shuffle", True)

    num_epochs = params.pop_int("num_epochs", 20)

    cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
    grad_norm = params.pop_float("grad_norm", None)
    grad_clipping = params.pop_float("grad_clipping", None)
    lr_scheduler_params = params.pop("learning_rate_scheduler", None)
    momentum_scheduler_params = params.pop("momentum_scheduler", None)

    validation_data_path = params.pop("validation_data_path", None)
    validation_prediction_path = params.pop("validation_prediction_path", None)

    warmup_epochs = params.pop("warmup_epochs", 0) 

    if isinstance(cuda_device, list):
        model_device = cuda_device[0]
    else:
        model_device = cuda_device
    if model_device >= 0:
        # Moving model to GPU here so that the optimizer state gets constructed on
        # the right device.
        model = model.cuda(model_device)

    bert_optim_params = params.pop("bert_optimizer", None)
    bert_name = "_bert_encoder"

    if bert_optim_params is not None:
        tune_after_layer_num = params.pop("bert_tune_layer", 12)

        frozen_regex_str = ["(_bert_encoder\.bert_model\.embeddings.*)",
                            "(_bert_encoder\.bert_model\.pooler.*)"]
        tune_regex_str = []
        for i in range(0, 12):
            # match all numbers greater than layer num via disjunction 
            tune_regex_one = f"({bert_name}\.bert_model\.encoder\.layer\.{i}\..*)"
            if i >= tune_after_layer_num:
                tune_regex_str.append(tune_regex_one)
            else:
                frozen_regex_str.append(tune_regex_one)
        tune_regex = re.compile("|".join(tune_regex_str))
        frozen_regex = re.compile("|".join(frozen_regex_str))
        # decide which params require grad for which optimizer 
        all_names = [n for n, p in model.named_parameters()]
        tune_bert_names = [n for n in all_names if tune_regex.match(n) is not None]
        frozen_names = [n for n in all_names if frozen_regex.match(n) is not None]
        # assert that they're disjoint
        assert(len(set(frozen_names) & set(tune_bert_names)) == 0)
        # set tunable params to require gradient, frozen ones to not require 
        for i, (n, p) in enumerate(model.named_parameters()):
            if n in frozen_names:
                p.requires_grad = False 
            else:
                p.requires_grad = True

        # extract BERT 
        bert_params = [[n, p] for n, p in model.named_parameters() if p.requires_grad and n in tune_bert_names]
        # make sure this matches the tuneable bert params 
        assert([x[0] for x in bert_params] == tune_bert_names)
        bert_optimizer = Optimizer.from_params(bert_params, bert_optim_params)
    else:
        # freeze all BERT params 
        tune_bert_names = []
        bert_optimizer = None 
        for i, (n, p) in enumerate(model.named_parameters()):
            if "_bert_encoder" in n:
                p.requires_grad = False 

    # model params 
    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad and n not in tune_bert_names]
    optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
    if "moving_average" in params:
        moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
    else:
        moving_average = None

    if lr_scheduler_params:
        lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
    else:
        lr_scheduler = None
    if momentum_scheduler_params:
        momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
    else:
        momentum_scheduler = None

    if 'checkpointer' in params:
        if 'keep_serialized_model_every_num_seconds' in params or \
                'num_serialized_models_to_keep' in params:
            raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods.")
        checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
    else:
        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)
    model_save_interval = params.pop_float("model_save_interval", None)
    summary_interval = params.pop_int("summary_interval", 100)
    histogram_interval = params.pop_int("histogram_interval", None)
    should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
    should_log_learning_rate = params.pop_bool("should_log_learning_rate", True)
    log_batch_size_period = params.pop_int("log_batch_size_period", None)
    accumulate_batches = params.pop("accumulate_batches", 1) 
    do_train_metrics = params.pop("do_train_metrics", False)
    params.assert_empty(cls.__name__)
    return cls(model=model,
               optimizer=optimizer,
               bert_optimizer=bert_optimizer,
               iterator=iterator,
               train_dataset=train_data,
               validation_dataset=validation_data,
               validation_data_path=validation_data_path,
               validation_prediction_path=validation_prediction_path,
               warmup_epochs = warmup_epochs, 
               patience=patience,
               validation_metric=validation_metric,
               validation_iterator=validation_iterator,
               shuffle=shuffle,
               num_epochs=num_epochs,
               serialization_dir=serialization_dir,
               cuda_device=cuda_device,
               grad_norm=grad_norm,
               grad_clipping=grad_clipping,
               learning_rate_scheduler=lr_scheduler,
               momentum_scheduler=momentum_scheduler,
               checkpointer=checkpointer,
               model_save_interval=model_save_interval,
               summary_interval=summary_interval,
               histogram_interval=histogram_interval,
               should_log_parameter_statistics=should_log_parameter_statistics,
               should_log_learning_rate=should_log_learning_rate,
               log_batch_size_period=log_batch_size_period,
               moving_average=moving_average,
               accumulate_batches=accumulate_batches,
               do_train_metrics=do_train_metrics) 
               

