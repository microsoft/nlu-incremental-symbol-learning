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

from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
#from miso.data.iterators.data_iterator import DecompDataIterator, DecompBasicDataIterator 
from miso.metrics.s_metric.s_metric import S, compute_s_metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("decomp_parsing")
class DecompTrainer(Trainer):

    def __init__(self,
                 validation_data_path: str,
                 validation_prediction_path: str,
                 semantics_only: bool,
                 drop_syntax: bool,
                 include_attribute_scores: bool = False,
                 warmup_epochs: int = 0,
                 syntactic_method:str = None,
                 accumulate_batches: int = 1,
                 bert_optimizer: Optimizer = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_data_path = validation_data_path
        self.validation_prediction_path = validation_prediction_path
        self.semantics_only=semantics_only
        self.drop_syntax=drop_syntax
        self.include_attribute_scores=include_attribute_scores
        self.accumulate_batches = accumulate_batches
        self.bert_optimizer = bert_optimizer

        self._warmup_epochs = warmup_epochs
        self._curr_epoch = 0

    def _update_validation_s_score(self, pred_instances: List[Dict[str, numpy.ndarray]],
                                         true_instances):
        """Write the validation output in pkl format, and compute the S score."""
        logger.info("Computing S")

        for batch in true_instances:
            assert(len(batch) == 1)

        true_graphs = [true_inst for batch in true_instances for true_inst in batch[0]['graph'] ]
        true_sents = [true_inst for batch in true_instances for true_inst in batch[0]['src_tokens_str']]

        pred_graphs = [DecompGraph.from_prediction(pred_inst) for pred_inst in pred_instances]

        ret = compute_s_metric(true_graphs, pred_graphs, true_sents, 
                               self.semantics_only, 
                               self.drop_syntax, 
                               self.include_attribute_scores)

        self.model.val_s_precision = float(ret[0]) * 100
        self.model.val_s_recall = float(ret[1]) * 100
        self.model.val_s_f1 = float(ret[2]) * 100

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

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss and updates loss weight. 
        Returns it and the number of batches.
        """
        # update loss weight 
        if hasattr(self.model, "loss_mixer") and self.model.loss_mixer is not None:
            self.model.loss_mixer.update_weights(
                                curr_epoch = self._curr_epoch, 
                                total_epochs = self._num_epochs
                                                )

        if self._curr_epoch < self._warmup_epochs:
            # skip the validation step for the warmup period 
            # this greatly reduces train time, since much of it is spent in validation 
            # with non-viable models 
            self._curr_epoch += 1
            return -1, -1

        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        # Disable multiple gpus in validation.
        num_gpus = 1

        raw_val_generator = val_iterator(self._validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data)/num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        val_true_instances  = []
        val_outputs: List[Dict[str, numpy.ndarray]] = []
        for batch_group in val_generator_tqdm:
            val_true_instances.append(batch_group)

            batch_output = self._validation_forward(batch_group)
            loss = batch_output.pop("loss", None)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

            # Update the validation outputs.
            peek = list(batch_output.values())[0]
            batch_size = peek.size(0) if isinstance(peek, torch.Tensor) else len(peek)
            instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in range(batch_size)]
            for name, value in batch_output.items():
                if isinstance(value, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if value.dim() == 0:
                        value = value.unsqueeze(0)
                    # shape: [batch_size, *]
                    value = value.detach().cpu().numpy()
                for instance_output, batch_element in zip(instance_separated_output, value):
                    instance_output[name] = batch_element
            val_outputs += instance_separated_output

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()
        self._update_validation_s_score(val_outputs, val_true_instances)

        if hasattr(self, "_update_validation_syntax_score"):
            self._update_validation_syntax_score(val_outputs, val_true_instances)

        return val_loss, batches_this_epoch

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

        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            #if batches_this_epoch % self.accumulate_batches == 0: 
            self.optimizer.zero_grad()
            if self.bert_optimizer is not None:
                self.bert_optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)
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
        print(params)
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
                 validation_iterator: DataIterator = None) -> DecompTrainer:
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

    semantics_only = params.pop("semantics_only", False)
    drop_syntax = params.pop("drop_syntax", True)
    include_attribute_scores = params.pop("include_attribute_scores", False)

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
    should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
    log_batch_size_period = params.pop_int("log_batch_size_period", None)
    syntactic_method = params.pop("syntactic_method", None)
    accumulate_batches = params.pop("accumulate_batches", 1) 

    params.assert_empty(cls.__name__)
    return cls(model=model,
               optimizer=optimizer,
               bert_optimizer=bert_optimizer,
               iterator=iterator,
               train_dataset=train_data,
               validation_dataset=validation_data,
               validation_data_path=validation_data_path,
               validation_prediction_path=validation_prediction_path,
               semantics_only=semantics_only,
               warmup_epochs = warmup_epochs, 
               syntactic_method = syntactic_method,
               drop_syntax=drop_syntax,
               include_attribute_scores=include_attribute_scores,
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
               accumulate_batches=accumulate_batches)
               

