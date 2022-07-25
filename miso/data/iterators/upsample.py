# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from typing import List, Tuple, Iterable, cast, Dict, Deque, Set
from collections import deque
import random
import logging
import pdb 
import json 
import pathlib 

from overrides import overrides
import numpy as np 

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.bucket_iterator import BucketIterator, sort_by_padding
from allennlp.data.iterators.data_iterator import DataIterator

from miso.data.dataset_readers.calflow_parsing.calflow_sequence import CalFlowSequence

@DataIterator.register("upsample")
class UpsampleIterator(BucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False,
                 fxn_of_interest: str = None,
                 upsample_factor: float = 1.0) -> None:
        super().__init__(sorting_keys=sorting_keys,
                        padding_noise=padding_noise,
                        biggest_batch_first=biggest_batch_first,
                        batch_size=batch_size,
                        instances_per_epoch=instances_per_epoch,
                        max_instances_in_memory=max_instances_in_memory,
                        cache_instances=cache_instances,
                        track_epoch=track_epoch,
                        maximum_samples_per_batch=maximum_samples_per_batch,
                        skip_smaller_batches=skip_smaller_batches)

        self.function_to_upsample = fxn_of_interest 
        self.upsample_factor = upsample_factor 


    def upsample(self, instance_list, function_to_upsample, upsample_factor):
        # split into interest and non-interest 
        interest, not_interest = [], []
        for instance in instance_list:
            if function_to_upsample in instance['tgt_tokens_str'].metadata: 
                interest.append(instance)
            else:
                not_interest.append(instance)

        num_to_sample = int(upsample_factor * len(interest)) - len(interest)

        interest_idxs = [i for i in range(len(interest))]
        additional_interest_idxs =  np.random.choice(interest_idxs, size=num_to_sample, replace=True).tolist()
        additional_interest = [interest[i] for i in additional_interest_idxs]
        interest += additional_interest
        instance_list = interest + not_interest 
        np.random.shuffle(instance_list)
        return instance_list

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # reset used indices each epoch 
        if hasattr(self, "used_idxs"): 
            self.used_idxs = set() 

        instances = self.upsample(instances, self.function_to_upsample, self.upsample_factor) 

        for instance_list in self._memory_sized_lists(instances):

            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
            evicted: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    if self._skip_smaller_batches and len(possibly_smaller_batches) < self._batch_size:
                        continue
                    # augment with minimal pairs 
                    batches.append(Batch(possibly_smaller_batches))
                    # put evicted instances in excess
                    #if evicted_instances is not None:
                    #    excess.extend(evicted_instances)
            if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches
@DataIterator.register("constant_ratio_upsample")
class ConstantRatioUpsampleIterator(BucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False,
                 fxn_of_interest: str = None,
                 upsample_ratio: float = 0.02) -> None:
        super().__init__(sorting_keys=sorting_keys,
                        padding_noise=padding_noise,
                        biggest_batch_first=biggest_batch_first,
                        batch_size=batch_size,
                        instances_per_epoch=instances_per_epoch,
                        max_instances_in_memory=max_instances_in_memory,
                        cache_instances=cache_instances,
                        track_epoch=track_epoch,
                        maximum_samples_per_batch=maximum_samples_per_batch,
                        skip_smaller_batches=skip_smaller_batches)

        self.function_to_upsample = fxn_of_interest 
        self.upsample_factor = upsample_ratio

    def upsample(self, instance_list, function_to_upsample, upsample_factor):
        # split into interest and non-interest 
        interest, not_interest = [], []
        for instance in instance_list:
            if function_to_upsample in instance['tgt_tokens_str'].metadata: 
                interest.append(instance)
            else:
                not_interest.append(instance)

        # num_to_sample = int(upsample_factor * len(interest)) - len(interest)
        num_to_sample = int(self.upsample_factor * len(instance_list))
        interest_idxs = [i for i in range(len(interest))]
        additional_interest_idxs =  np.random.choice(interest_idxs, size=num_to_sample, replace=True).tolist()
        additional_interest = [interest[i] for i in additional_interest_idxs]
        interest += additional_interest
        instance_list = interest + not_interest 
        np.random.shuffle(instance_list)
        return instance_list

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # reset used indices each epoch 
        if hasattr(self, "used_idxs"): 
            self.used_idxs = set() 

        instances = self.upsample(instances, self.function_to_upsample, self.upsample_factor) 

        for instance_list in self._memory_sized_lists(instances):

            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
            evicted: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    if self._skip_smaller_batches and len(possibly_smaller_batches) < self._batch_size:
                        continue
                    # augment with minimal pairs 
                    batches.append(Batch(possibly_smaller_batches))
                    # put evicted instances in excess
                    #if evicted_instances is not None:
                    #    excess.extend(evicted_instances)
            if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches

@DataIterator.register("constant_ratio_upsample_no_source")
class NoSourceConstantRatioUpsampleIterator(BucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False,
                 fxn_of_interest: str = None,
                 source_triggers: str = None,
                 upsample_ratio: float = 0.02) -> None:
        super().__init__(sorting_keys=sorting_keys,
                        padding_noise=padding_noise,
                        biggest_batch_first=biggest_batch_first,
                        batch_size=batch_size,
                        instances_per_epoch=instances_per_epoch,
                        max_instances_in_memory=max_instances_in_memory,
                        cache_instances=cache_instances,
                        track_epoch=track_epoch,
                        maximum_samples_per_batch=maximum_samples_per_batch,
                        skip_smaller_batches=skip_smaller_batches)

        self.function_to_upsample = fxn_of_interest 
        self.upsample_factor = upsample_ratio
        self.source_triggers = source_triggers.split(",")

    def has_source_triggers(self, instance): 
        source_tokens = instance['src_tokens_str'].metadata
        for trig in self.source_triggers:
            if trig in source_tokens:
                return True
        return False

    def upsample(self, instance_list, function_to_upsample, upsample_factor):
        # split into interest and non-interest 
        interest, not_interest = [], []
        for instance in instance_list:
            if function_to_upsample in instance['tgt_tokens_str'].metadata and not self.has_source_triggers(instance): 
                interest.append(instance)
            else:
                not_interest.append(instance)

        # num_to_sample = int(upsample_factor * len(interest)) - len(interest)
        num_to_sample = int(self.upsample_factor * len(instance_list))
        interest_idxs = [i for i in range(len(interest))]
        additional_interest_idxs =  np.random.choice(interest_idxs, size=num_to_sample, replace=True).tolist()
        additional_interest = [interest[i] for i in additional_interest_idxs]
        interest += additional_interest
        instance_list = interest + not_interest 
        np.random.shuffle(instance_list)
        return instance_list

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # reset used indices each epoch 
        if hasattr(self, "used_idxs"): 
            self.used_idxs = set() 

        instances = self.upsample(instances, self.function_to_upsample, self.upsample_factor) 

        for instance_list in self._memory_sized_lists(instances):

            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
            evicted: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    if self._skip_smaller_batches and len(possibly_smaller_batches) < self._batch_size:
                        continue
                    # augment with minimal pairs 
                    batches.append(Batch(possibly_smaller_batches))
                    # put evicted instances in excess
                    #if evicted_instances is not None:
                    #    excess.extend(evicted_instances)
            if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches
