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

#np.random.seed(12)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class MinimalPairIterator(BucketIterator):
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
                 skip_smaller_batches: bool = False) -> None:
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

    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real or synthetic data
        """
        raise NotImplementedError
        
    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # reset used indices each epoch 
        if hasattr(self, "used_idxs"): 
            self.used_idxs = set() 

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
                    possibly_smaller_batches, evicted_instances = self.create_minimal_pairs(possibly_smaller_batches)
                    batches.append(Batch(possibly_smaller_batches))
                    # put evicted instances in excess
                    #if evicted_instances is not None:
                    #    excess.extend(evicted_instances)
            if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                excess, __ = self.create_minimal_pairs(excess)
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

@DataIterator.register("synthetic_min_pair")
class SyntheticMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 dataset_reader: DatasetReader = None) -> None:
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
        self.dataset_reader = dataset_reader

    def choose_tgt_token(self):
        return "Func1"
    def choose_src_token(self):
        return "a"

    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for synthetic data
        """
        # copy instances 
        output_instances = [x for x in instances]
        evictable_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_inputs'].metadata.strip().split(" ")
            # don't need minimal pair for anything that's length 1 
            if len(tgt_tokens) == 1: 
                evictable_instances.append(i)
                continue
            # don't need minimal pairs for anything that's not fxn_of_interest 
            if self.fxn_of_interest not in tgt_tokens:
                evictable_instances.append(i)
                continue

            # take from [1:] to remove __User token 
            src_tokens = inst['src_tokens_str'].metadata[1:]
            tgt_index = tgt_tokens.index(self.fxn_of_interest)
            tgt_min_pair = [tok for tok in tgt_tokens]
            src_min_pair = [tok for tok in src_tokens]
            tgt_min_pair[tgt_index] = self.choose_tgt_token()
            src_min_pair[tgt_index] = self.choose_src_token()

            assert(len(src_min_pair) == len(tgt_min_pair))
            assert(self.fxn_of_interest not in tgt_min_pair)

            new_src_str = " ".join(src_min_pair)
            new_tgt_str = " ".join(tgt_min_pair)
            new_sequence = CalFlowSequence(new_src_str, new_tgt_str, fxn_of_interest=self.fxn_of_interest) 
            new_instance = self.dataset_reader.text_to_instance(new_sequence)
            # if we can't evict any instances, then everything is function of interest or it's the first input 
            # skip it for now, probably very rare 
            if len(evictable_instances) == 0:
                continue
            instance_to_evict = random.choice(evictable_instances)
            # evict a current instance to make sure the number of training examples stays the same 
            output_instances[instance_to_evict] = new_instance

        return output_instances, None

@DataIterator.register("synthetic_exclude_min_pair")
class SyntheticExcludeMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 dataset_reader: DatasetReader = None) -> None:
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
        self.dataset_reader = dataset_reader

    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for synthetic data
        """
        # copy instances 
        output_instances = [x for x in instances]

        instances_of_interest = [] 
        lens_of_interest = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_inputs'].metadata.strip().split(" ")
            if self.fxn_of_interest in tgt_tokens:
                instances_of_interest.append(i) 
                lens_of_interest.append(len(tgt_tokens))

        for i, inst in enumerate(instances):
            # skip b instances 
            if i in instances_of_interest:
                continue

            tgt_tokens = inst['tgt_tokens_inputs'].metadata.strip().split(" ")
            if self.fxn_of_interest not in tgt_tokens and len(tgt_tokens) in lens_of_interest:
                # if same len but no 'b', then minimal pair, remove it from the batch 
                # choose a new length that isn't the same and create an instance
                allowable_lens = list(set([i for i in range(1, 14)]) - set([len(tgt_tokens)]))
                new_len = np.random.choice(allowable_lens)
                new_src_str = " ".join(['a' for __ in range(new_len)])
                new_tgt_str = " ".join(['Func1' for __ in range(new_len)])
                new_sequence = CalFlowSequence(new_src_str, new_tgt_str, fxn_of_interest=self.fxn_of_interest) 
                new_instance = self.dataset_reader.text_to_instance(new_sequence)
                output_instances[i] = new_instance
            else:
                # if not a minimal pair, keep it 
                output_instances[i] = inst

        return output_instances, None


@DataIterator.register("real_min_pair")
class RealMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 dataset_reader: DatasetReader = None) -> None:
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
        self.dataset_reader = dataset_reader

        self.lookup_table = json.load(open(pair_lookup_table))

    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # copy instances 
        output_instances = [x for x in instances]
        evictable_instances = []
        evicted_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_str'].metadata[1:-1]
            inst_index = str(inst['line_index'].metadata)
            # don't need minimal pairs for anything that's not fxn_of_interest 
            if self.fxn_of_interest not in tgt_tokens:
                evictable_instances.append(i)
                continue
            
            
            new_src_str, new_tgt_str = self.lookup_table[inst_index]
            #pdb.set_trace()

            new_graph = CalFlowGraph(new_src_str, 
                                     new_tgt_str, 
                                     use_agent_utterance = self.dataset_reader.use_agent_utterance, 
                                     use_context = self.dataset_reader.use_context,
                                     use_program = self.dataset_reader.use_program,
                                     fxn_of_interest=self.fxn_of_interest) 
            new_instance = self.dataset_reader.text_to_instance(new_graph)
            # if we can't evict any instances, then everything is function of interest or it's the first input 
            # skip it for now, probably very rare 
            if len(evictable_instances) == 0:
                continue

            instance_to_evict = random.choice(evictable_instances)
            # evict a current instance to make sure the number of training examples stays the same 
            evicted_instances.append(instances[instance_to_evict])
            output_instances[instance_to_evict] = new_instance

        return output_instances, evicted_instances
        
@DataIterator.register("exclude_real_min_pair")
class ExcludeRealMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 dataset_reader: DatasetReader = None) -> None:
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
        self.dataset_reader = dataset_reader

        self.lookup_table = json.load(open(pair_lookup_table))

    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # TODO (elias)
        raise NotImplementedError

@DataIterator.register("generated_real_min_pair")
class GeneratedRealMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 train_path: str = None, 
                 dataset_reader: DatasetReader = None) -> None:
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
        self.dataset_reader = dataset_reader

        self.lookup_table = json.load(open(pair_lookup_table))
        self.train_path = pathlib.Path(train_path)

        src_path = self.train_path.joinpath("train.src_tok") 
        idx_path = self.train_path.joinpath("train.idx") 
        tgt_path = self.train_path.joinpath("train.tgt")

        train_src = [line.strip() for line in open(src_path).readlines()]
        train_tgt = [line.strip() for line in open(tgt_path).readlines()]
        train_idx = [line.strip() for line in open(idx_path).readlines()]

        self.train_idx_lookup = {int(idx): (src, tgt) for src, idx, tgt in zip(train_src, train_idx, train_tgt)}

        self.used_idxs = set()

    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # copy instances 
        output_instances = [x for x in instances]
        evictable_instances = []
        evicted_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_str'].metadata[1:-1]
            inst_index = str(inst['line_index'].metadata)
            # don't need minimal pairs for anything that's not fxn_of_interest 
            if self.fxn_of_interest not in tgt_tokens:
                evictable_instances.append(i)
                continue

            new_idx = None
            j = 0
            skip = False
            # cycle through candidate indices until find an unused one 
            while new_idx is None:
                try:
                    candidate = self.lookup_table[inst_index][j]
                except IndexError:
                    skip = True
                    break
                j+=1
                if candidate in self.used_idxs:
                    continue
                new_idx = candidate
            if skip:
                continue
            self.lookup_table[inst_index] = self.lookup_table[inst_index][1:]
            self.used_idxs.add(new_idx)

            new_src_str, new_tgt_str = self.train_idx_lookup[new_idx]
            src_toks = [str(x) for x in inst['source_tokens'].tokens]
            user_idxs = [i for i, x in enumerate(src_toks) if x == "__User"]
            src_str = " ".join(src_toks[user_idxs[-1]+1:])
            #print(f"Pairing {src_str} with {new_src_str}")
            #pdb.set_trace()
            new_graph = CalFlowGraph(new_src_str, 
                                     new_tgt_str, 
                                     use_agent_utterance = self.dataset_reader.use_agent_utterance, 
                                     use_context = self.dataset_reader.use_context,
                                     use_program = self.dataset_reader.use_program,
                                     fxn_of_interest=self.fxn_of_interest) 
            new_instance = self.dataset_reader.text_to_instance(new_graph)
            # if we can't evict any instances, then everything is function of interest or it's the first input 
            # skip it for now, probably very rare 
            if len(evictable_instances) == 0:
                continue

            instance_to_evict = random.choice(evictable_instances)
            # evict a current instance to make sure the number of training examples stays the same 
            evicted_instances.append(instances[instance_to_evict])
            output_instances[instance_to_evict] = new_instance

        return output_instances, evicted_instances


@DataIterator.register("full_generated_min_pair")
class FullMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 train_path: str = None, 
                 dataset_reader: DatasetReader = None,
                 choose_top: bool = True,
                 sample_top_k: int = -1) -> None:

        # divide batch size for pairing 
        batch_size = int(batch_size/2)
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
        self.dataset_reader = dataset_reader

        self.lookup_table = json.load(open(pair_lookup_table))
        self.train_path = pathlib.Path(train_path)

        src_path = self.train_path.joinpath("train.src_tok") 
        idx_path = self.train_path.joinpath("train.idx") 
        tgt_path = self.train_path.joinpath("train.tgt")

        train_src = [line.strip() for line in open(src_path).readlines()]
        train_tgt = [line.strip() for line in open(tgt_path).readlines()]
        train_idx = [line.strip() for line in open(idx_path).readlines()]

        self.train_idx_lookup = {int(idx): (src, tgt) for src, idx, tgt in zip(train_src, train_idx, train_tgt)}
        self.used_idxs = set()

        # if choosing top, just take the best possible pair 
        self.choose_top = choose_top
        # if sampling top k, sample randomly from the top k minimal pairs 
        self.sample_top_k = sample_top_k

        if self.choose_top and self.sample_top_k > -1: 
            raise AssertionError(f"You cannot have choose_top and sample_top_k set simultaneously, you must pick one method.")
        if not self.choose_top and self.sample_top_k == -1: 
            raise AssertionError(f"You cannot have choose_top and sample_top_k unset simultaneously, you must pick one method.")

    def get_topk_choices(self, ranking): 
        ranking = [x for x in ranking if int(x) not in self.used_idxs]
        if len(ranking) == 0:
            return None
        ranking = ranking[0: self.sample_top_k]
        return ranking 


    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # copy instances 
        #output_instances = [x for x in instances]
        paired = 0
        skipped_original_instances = 0
        skipped_by_end = 0
        output_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_str'].metadata[1:-1]
            inst_index = str(inst['line_index'].metadata)
            # if the instance has been used as a pair before, skip it 
            if int(inst_index) in self.used_idxs:
                skipped_original_instances += 1
                # need to remove from instances 
                continue 
            output_instances.append(inst)
            # add instance idx so that instance is not seen as a pair in the future 
            self.used_idxs.add(int(inst_index))
            new_idx = None
            j = 0
            skip = False
            # cycle through candidate indices until find an unused one 
            while new_idx is None:
                try:
                    if self.choose_top:
                        # choose the highest-ranked instance
                        candidate = self.lookup_table[inst_index][j]
                    else:
                        # choose a random instance from the top k
                        #print(f"choosing from {len(self.lookup_table[inst_index][0:self.sample_top_k])}")
                        choices = self.get_topk_choices(self.lookup_table[inst_index])
                        if choices is None or len(choices) == 0:
                            #print(f"skipping!")
                            skip = True
                            break
                        elif len(choices) == 1:
                            choice_idx = choices[0]
                        else:
                            choice_idx = np.random.choice(len(choices)-1)
                        candidate = choices[choice_idx]
                        #print(f"chose {candidate}")
                except IndexError:
                    skip = True
                    break
                j+=1
                # if the candidate has been seen before, skip it 
                if int(candidate) in self.used_idxs:
                    continue

                new_idx = candidate
            if skip:
                skipped_by_end += 1
                continue
            #self.lookup_table[inst_index] = self.lookup_table[inst_index][1:]
            self.used_idxs.add(int(new_idx))

            new_src_str, new_tgt_str = self.train_idx_lookup[new_idx]
            src_toks = [str(x) for x in inst['source_tokens'].tokens]
            user_idxs = [i for i, x in enumerate(src_toks) if x == "__User"]
            src_str = " ".join(src_toks[user_idxs[-1]+1:])
            #print(f"Pairing {src_str} with {new_src_str}")
            #pdb.set_trace()
            new_graph = CalFlowGraph(new_src_str, 
                                     new_tgt_str, 
                                     use_agent_utterance = self.dataset_reader.use_agent_utterance, 
                                     use_context = self.dataset_reader.use_context,
                                     use_program = self.dataset_reader.use_program,
                                     fxn_of_interest=self.fxn_of_interest) 
            new_instance = self.dataset_reader.text_to_instance(new_graph)
            paired += 1
            output_instances.append(new_instance)

        #print(f"SIZE OF USED IDXS {len(self.used_idxs)}")
        #print(f"SIZE OF USED IDXS {self.used_idxs}")
        #print(f"Length of batch is {len(output_instances)}")
        #print(f"Skipped instances: {skipped_original_instances}, Skipped by backoff: {skipped_by_end}, Paired: {paired}")
        return output_instances, None

@DataIterator.register("random_baseline_min_pair")
class RandomBaselineMinimalPairIterator(GeneratedRealMinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 train_path: str = None, 
                 dataset_reader: DatasetReader = None) -> None:
        super().__init__(sorting_keys=sorting_keys,
                        padding_noise=padding_noise,
                        biggest_batch_first=biggest_batch_first,
                        batch_size=batch_size,
                        instances_per_epoch=instances_per_epoch,
                        max_instances_in_memory=max_instances_in_memory,
                        cache_instances=cache_instances,
                        track_epoch=track_epoch,
                        maximum_samples_per_batch=maximum_samples_per_batch,
                        skip_smaller_batches=skip_smaller_batches,
                        fxn_of_interest=fxn_of_interest,
                        pair_lookup_table=pair_lookup_table,
                        train_path=train_path,
                        dataset_reader=dataset_reader)


    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # copy instances 
        output_instances = [x for x in instances]
        evictable_instances = []
        evicted_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_str'].metadata[1:-1]
            inst_index = str(inst['line_index'].metadata)
            # don't need minimal pairs for anything that's not fxn_of_interest 
            if self.fxn_of_interest not in tgt_tokens:
                evictable_instances.append(i)
                continue

            new_idx = None
            j = 0
            skip = False
            # pick a random training point for each FindManager example 

            valid_indices = set(self.train_idx_lookup.keys()) - self.used_idxs - set([int(inst_index)])
            new_idx = np.random.choice(list(valid_indices))
            self.used_idxs.add(new_idx)
            self.used_idxs.add(int(inst_index))

            new_src_str, new_tgt_str = self.train_idx_lookup[new_idx]
            src_toks = [str(x) for x in inst['source_tokens'].tokens]
            user_idxs = [i for i, x in enumerate(src_toks) if x == "__User"]
            src_str = " ".join(src_toks[user_idxs[-1]+1:])
            #print(f"Pairing {src_str} with {new_src_str}")
            #pdb.set_trace()
            new_graph = CalFlowGraph(new_src_str, 
                                     new_tgt_str, 
                                     use_agent_utterance = self.dataset_reader.use_agent_utterance, 
                                     use_context = self.dataset_reader.use_context,
                                     use_program = self.dataset_reader.use_program,
                                     fxn_of_interest=self.fxn_of_interest) 
            new_instance = self.dataset_reader.text_to_instance(new_graph)
            # if we can't evict any instances, then everything is function of interest or it's the first input 
            # skip it for now, probably very rare 
            if len(evictable_instances) == 0:
                continue

            instance_to_evict = random.choice(evictable_instances)
            # evict a current instance to make sure the number of training examples stays the same 
            evicted_instances.append(instances[instance_to_evict])
            output_instances[instance_to_evict] = new_instance

        return output_instances, evicted_instances

@DataIterator.register("full_generated_min_pair_temperature")
class TemperatureMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 train_path: str = None, 
                 dataset_reader: DatasetReader = None,
                 choose_top: bool = True,
                 sample_top_k: int = -1) -> None:

        # TODO (Elias) 
        raise NotImplementedError
        # divide batch size for pairing 
        batch_size = int(batch_size/2)
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
        self.dataset_reader = dataset_reader

        self.lookup_table = json.load(open(pair_lookup_table))
        self.train_path = pathlib.Path(train_path)

        src_path = self.train_path.joinpath("train.src_tok") 
        idx_path = self.train_path.joinpath("train.idx") 
        tgt_path = self.train_path.joinpath("train.tgt")

        train_src = [line.strip() for line in open(src_path).readlines()]
        train_tgt = [line.strip() for line in open(tgt_path).readlines()]
        train_idx = [line.strip() for line in open(idx_path).readlines()]

        self.train_idx_lookup = {int(idx): (src, tgt) for src, idx, tgt in zip(train_src, train_idx, train_tgt)}
        self.used_idxs = set()

        # if choosing top, just take the best possible pair 
        self.choose_top = choose_top
        # if sampling top k, sample randomly from the top k minimal pairs 
        self.sample_top_k = sample_top_k

        if self.choose_top and self.sample_top_k > -1: 
            raise AssertionError(f"You cannot have choose_top and sample_top_k set simultaneously, you must pick one method.")
        if not self.choose_top and self.sample_top_k == -1: 
            raise AssertionError(f"You cannot have choose_top and sample_top_k unset simultaneously, you must pick one method.")

    def get_topk_choices(self, ranking): 
        ranking = [x for x in ranking if int(x) not in self.used_idxs]
        if len(ranking) == 0:
            return None
        ranking = ranking[0: self.sample_top_k]
        return ranking 


    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # copy instances 
        #output_instances = [x for x in instances]
        paired = 0
        skipped_original_instances = 0
        skipped_by_end = 0
        output_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_str'].metadata[1:-1]
            inst_index = str(inst['line_index'].metadata)
            # if the instance has been used as a pair before, skip it 
            if int(inst_index) in self.used_idxs:
                skipped_original_instances += 1
                # need to remove from instances 
                continue 
            output_instances.append(inst)
            # add instance idx so that instance is not seen as a pair in the future 
            self.used_idxs.add(int(inst_index))
            new_idx = None
            j = 0
            skip = False
            # cycle through candidate indices until find an unused one 
            while new_idx is None:
                try:
                    if self.choose_top:
                        # choose the highest-ranked instance
                        candidate = self.lookup_table[inst_index][j]
                    else:
                        # choose a random instance from the top k
                        #print(f"choosing from {len(self.lookup_table[inst_index][0:self.sample_top_k])}")
                        choices = self.get_topk_choices(self.lookup_table[inst_index])
                        if choices is None or len(choices) == 0:
                            print(f"skipping!")
                            skip = True
                            break
                        elif len(choices) == 1:
                            choice_idx = choices[0]
                        else:
                            choice_idx = np.random.choice(len(choices)-1)
                        candidate = choices[choice_idx]
                        #print(f"chose {candidate}")
                except IndexError:
                    skip = True
                    break
                j+=1
                # if the candidate has been seen before, skip it 
                if int(candidate) in self.used_idxs:
                    continue

                new_idx = candidate
            if skip:
                skipped_by_end += 1
                continue
            #self.lookup_table[inst_index] = self.lookup_table[inst_index][1:]
            self.used_idxs.add(int(new_idx))

            new_src_str, new_tgt_str = self.train_idx_lookup[new_idx]
            src_toks = [str(x) for x in inst['source_tokens'].tokens]
            user_idxs = [i for i, x in enumerate(src_toks) if x == "__User"]
            src_str = " ".join(src_toks[user_idxs[-1]+1:])
            #print(f"Pairing {src_str} with {new_src_str}")
            #pdb.set_trace()
            new_graph = CalFlowGraph(new_src_str, 
                                     new_tgt_str, 
                                     use_agent_utterance = self.dataset_reader.use_agent_utterance, 
                                     use_context = self.dataset_reader.use_context,
                                     use_program = self.dataset_reader.use_program,
                                     fxn_of_interest=self.fxn_of_interest) 
            new_instance = self.dataset_reader.text_to_instance(new_graph)
            paired += 1
            output_instances.append(new_instance)

        #print(f"SIZE OF USED IDXS {len(self.used_idxs)}")
        #print(f"SIZE OF USED IDXS {self.used_idxs}")
        #print(f"Length of batch is {len(output_instances)}")
        #print(f"Skipped instances: {skipped_original_instances}, Skipped by backoff: {skipped_by_end}, Paired: {paired}")
        return output_instances, None


@DataIterator.register("low_resource_min_pair")
class LowResourceMinimalPairIterator(MinimalPairIterator):
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
                 fxn_of_interest: str = "Func2",
                 pair_lookup_table: str = None,
                 train_path: str = None, 
                 dataset_reader: DatasetReader = None,
                 choose_top: bool = True,
                 sample_top_k: int = -1,
                 percentile: float = 0.5,
                 frequency_path: str = None) -> None:


        # likely this will make batches smaller than they need to be  
        # since if we're looking at the bottom 50% of functions by frequency, 
        # they will appear in < 50% of programs 
        batch_size = int(batch_size * percentile)

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
        self.dataset_reader = dataset_reader

        self.frequency_data = json.load(open(frequency_path))
        self.low_freq_fxns = self.get_low_freq(percentile)


        self.lookup_table = json.load(open(pair_lookup_table))
        self.train_path = pathlib.Path(train_path)

        src_path = self.train_path.joinpath("train.src_tok") 
        idx_path = self.train_path.joinpath("train.idx") 
        tgt_path = self.train_path.joinpath("train.tgt")

        train_src = [line.strip() for line in open(src_path).readlines()]
        train_tgt = [line.strip() for line in open(tgt_path).readlines()]
        train_idx = [line.strip() for line in open(idx_path).readlines()]

        self.train_idx_lookup = {int(idx): (src, tgt) for src, idx, tgt in zip(train_src, train_idx, train_tgt)}
        self.used_idxs = set()

        # if choosing top, just take the best possible pair 
        self.choose_top = choose_top
        # if sampling top k, sample randomly from the top k minimal pairs 
        self.sample_top_k = sample_top_k

        if self.choose_top and self.sample_top_k > -1: 
            raise AssertionError(f"You cannot have choose_top and sample_top_k set simultaneously, you must pick one method.")
        if not self.choose_top and self.sample_top_k == -1: 
            raise AssertionError(f"You cannot have choose_top and sample_top_k unset simultaneously, you must pick one method.")

    def get_low_freq(self, percentile):
        sorted_freq_items = sorted(self.frequency_data.items(), key=lambda x: x[1])
        idx = int(percentile * len(sorted_freq_items)) + 1 
        under_percentile = [x[0] for x in sorted_freq_items][0:idx]
        return under_percentile

    def get_topk_choices(self, ranking): 
        ranking = [x for x in ranking if int(x) not in self.used_idxs]
        if len(ranking) == 0:
            return None
        ranking = ranking[0: self.sample_top_k]
        return ranking 

    def is_low_resource(self, tgt_tokens): 
        return any([x in self.low_freq_fxns for x in tgt_tokens])

    @overrides
    def create_minimal_pairs(self, instances: Iterable[Instance]):
        """
        Function to create minimal pairs for real data
        """
        # copy instances 
        #output_instances = [x for x in instances]
        paired = 0
        skipped_original_instances = 0
        skipped_by_end = 0
        output_instances = []
        for i, inst in enumerate(instances):
            tgt_tokens = inst['tgt_tokens_str'].metadata[1:-1]
            is_lr = self.is_low_resource(tgt_tokens)
            inst_index = str(inst['line_index'].metadata)
            # if the instance has been used as a pair before, skip it 
            if int(inst_index) in self.used_idxs:
                skipped_original_instances += 1
                # need to remove from instances 
                continue 
            output_instances.append(inst)
            # add instance idx so that instance is not seen as a pair in the future 
            self.used_idxs.add(int(inst_index))
            new_idx = None
            j = 0
            skip = False

            # only add min pair for low resource functions 
            if is_lr:
                # cycle through candidate indices until find an unused one 
                while new_idx is None:
                    try:
                        if self.choose_top:
                            # choose the highest-ranked instance
                            candidate = self.lookup_table[inst_index][j]
                        else:
                            # choose a random instance from the top k
                            #print(f"choosing from {len(self.lookup_table[inst_index][0:self.sample_top_k])}")
                            choices = self.get_topk_choices(self.lookup_table[inst_index])
                            if choices is None or len(choices) == 0:
                                print(f"skipping!")
                                skip = True
                                break
                            elif len(choices) == 1:
                                choice_idx = choices[0]
                            else:
                                choice_idx = np.random.choice(len(choices)-1)
                            candidate = choices[choice_idx]
                            #print(f"chose {candidate}")
                    except IndexError:
                        skip = True
                        break
                    j+=1
                    # if the candidate has been seen before, skip it 
                    if int(candidate) in self.used_idxs:
                        continue

                    new_idx = candidate
                if skip:
                    skipped_by_end += 1
                    continue
                #self.lookup_table[inst_index] = self.lookup_table[inst_index][1:]
                self.used_idxs.add(int(new_idx))

                new_src_str, new_tgt_str = self.train_idx_lookup[new_idx]
                src_toks = [str(x) for x in inst['source_tokens'].tokens]
                user_idxs = [i for i, x in enumerate(src_toks) if x == "__User"]
                src_str = " ".join(src_toks[user_idxs[-1]+1:])
                print(f"Pairing {src_str} with {new_src_str}")
                print(f"Pairing {' '.join(tgt_tokens)} with {new_tgt_str}")
                #pdb.set_trace()
                new_graph = CalFlowGraph(new_src_str, 
                                        new_tgt_str, 
                                        use_agent_utterance = self.dataset_reader.use_agent_utterance, 
                                        use_context = self.dataset_reader.use_context,
                                        use_program = self.dataset_reader.use_program,
                                        fxn_of_interest=self.fxn_of_interest) 
                new_instance = self.dataset_reader.text_to_instance(new_graph)
                paired += 1
                output_instances.append(new_instance)

        return output_instances, None