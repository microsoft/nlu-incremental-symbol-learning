# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from intent.source_lookup import get_max_probs
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from typing import List, Tuple, Iterable, cast, Dict, Deque, Set
from collections import deque, defaultdict
import random
import logging
import pdb 
import json 
import pathlib 

from overrides import overrides
import numpy as np 

FUNCTION = ['the','and','a','an','this','these', "__user", "__agent", "with", "?", ".", "!", ","]
np.random.seed(12)

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.util import lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.bucket_iterator import BucketIterator, sort_by_padding
from allennlp.data.iterators.data_iterator import DataIterator

from miso.data.dataset_readers.calflow_parsing.calflow_sequence import CalFlowSequence

@DataIterator.register("filter_by_source")
class FilterIterator(BucketIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 top_k: int = 3,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False,
                 fxn_of_interest: str = None) -> None:
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

        self.fxn_of_interest = fxn_of_interest 
        self.top_k = top_k 


    def get_probs(self, instances, exclude=True, exclude_function=False):
        count_word = defaultdict(int)
        count_fxn = defaultdict(int)
        count_fxn_and_word = defaultdict(lambda: defaultdict(int))
        count_of_word_and_intent = defaultdict(lambda: defaultdict(int))
        for ex in instances:
            output_strs = ex['tgt_tokens_str'].metadata 
            line = ex['src_tokens_str'].metadata
            user_idxs = [j for j, tok in enumerate(line) if tok == "__User"]
            line = line[user_idxs[-1]+1:]
            input_strs  = [x.lower() for x in line]

            for fxn in set(output_strs):
                count_fxn[fxn] += 1
                for tok in set(input_strs):
                    count_word[tok] += 1
                    count_fxn_and_word[fxn][tok] += 1
                    #count_of_word_and_intent[fxn][tok] +=1 

        prob_fxn_given_word = defaultdict(lambda: defaultdict(int))
        prob_word_given_fxn = defaultdict(lambda: defaultdict(int))
        for fxn in count_fxn_and_word.keys():
            for word in count_fxn_and_word[fxn].keys():
                # Exclude anything that happens less than 10% of the time 
                if exclude and count_fxn_and_word[fxn][word] / count_fxn[fxn] < 0.10: 
                    continue 
                if exclude_function and word.lower() in FUNCTION:
                    continue

                single_prob_fxn_given_word = count_fxn_and_word[fxn][word] / count_word[word]
                prob_fxn_given_word[fxn][word] = single_prob_fxn_given_word
                single_prob_word_given_fxn = count_fxn_and_word[fxn][word] / count_fxn[fxn]
                prob_word_given_fxn[fxn][word] = single_prob_word_given_fxn

        return prob_fxn_given_word, prob_word_given_fxn

    def get_max_probs(self, probs): 
        max_probs = {intent: None for intent in probs.keys()}
        for intent, prob_set in probs.items():
            top_k = sorted(list(prob_set.items()), key = lambda x: x[1])[-self.top_k:]
            max_probs[intent] = top_k
        return max_probs

    def filter(self, instances, top_k_words): 
        to_ret = []
        for inst in instances:
            line = inst['src_tokens_str'].metadata
            user_idxs = [j for j, tok in enumerate(line) if tok == "__User"]
            line = line[user_idxs[-1]+1:]
            input_strs  = [x.lower() for x in line]
            for word in top_k_words:
                if self.fxn_of_interest not in inst['tgt_tokens_str'].metadata and word in input_strs:
                    # print(f"skipping {line} for having {word} in {inst['tgt_tokens_str'].metadata}")
                    # skip 
                    continue 
                else:
                    to_ret.append(inst)
        return to_ret 


    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # reset used indices each epoch 
        if hasattr(self, "used_idxs"): 
            self.used_idxs = set() 

        prob_fxn_given_word, prob_word_given_fxn = self.get_probs(instances, exclude_function=True)
        # for speed, pair down to just one 
        probs = {self.fxn_of_interest: prob_fxn_given_word[self.fxn_of_interest]}
        top_k_words = self.get_max_probs(probs)[self.fxn_of_interest]
        #print(top_k_words)
        #pdb.set_trace()
        top_k_words = [x[0] for x in top_k_words]
        instances = self.filter(instances, top_k_words) 

        for instance_list in self._memory_sized_lists(instances):
            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
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
