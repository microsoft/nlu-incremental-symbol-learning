# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import List, Any, Tuple
import re 
import pdb 

from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
import numpy as np 

from miso.data.dataset_readers.decomp_parsing.decomp import SourceCopyVocabulary
from miso.data.dataset_readers.calflow_parsing.calflow_graph import NOBRACK, PROGRAM_SEP, PAD_EDGE
from dataflow.core.lispress import parse_lispress, render_compact


class CalFlowSequence:
    def __init__(self, 
                src_str: str,
                tgt_str: str,
                use_agent_utterance: bool = False, 
                use_context: bool = False,
                fxn_of_interest: str = None):
        #self.program = self.tgt_str_to_program(tgt_str)
        self.src_str = src_str.strip() 
        self.tgt_str = tgt_str
        self.use_agent_utterance = use_agent_utterance
        self.use_context = use_context
        self.fxn_of_interest = fxn_of_interest

    def get_list_data(self, 
                     bos: str = None, 
                     eos: str = None, 
                     bert_tokenizer = None, 
                     max_tgt_length: int = None):
        """
        Converts SMCalFlow graph into a linearized list of tokens and affiliated metadata 
        """

        # tokenize quotations for now so that source copy works on names 
        #tgt_str = re.sub('"(?=\w)', '" ', self.tgt_str)
        #tgt_str = re.sub('(?<=\w)"', ' "', tgt_str)
        tgt_tokens = self.tgt_str.strip().split(" ")
        mask = [1 for i in range(len(tgt_tokens))]

        def trim_very_long_tgt_tokens(tgt_tokens: List[str], 
                                      mask: List[int]) -> Tuple[List[str], List[int]]:
            tgt_tokens = tgt_tokens[:max_tgt_length]
            mask = mask[:max_tgt_length]
            return (tgt_tokens, mask)

        if max_tgt_length is not None:
            tgt_tokens, mask = trim_very_long_tgt_tokens(tgt_tokens, mask)

        if bos:
            tgt_tokens = [bos] + tgt_tokens
        if eos:
            tgt_tokens = tgt_tokens + [eos]

        tgt_indices = [i for i in range(len(tgt_tokens))]
        tgt_copy_indices = [0 for __ in range(len(tgt_tokens))]
        tgt_copy_map = [(i, copy_idx) for i, copy_idx in enumerate(tgt_indices)]


        prev_utt = None
        agent_utt = None
        user_utt = None

        if self.use_context: 
            src_string_split = re.split("(__User)|(__StartOfProgram)|(__Agent)", self.src_str) 
            src_string_split = [x for x in src_string_split if x != '' and x is not None]
            # just 2 cases
            # Utterance
            # Utterance, Program, Utterance 
            if len(src_string_split) == 3:
                user_utt = src_string_split[1].strip()
            elif len(src_string_split) == 7: 
                prev_utt = src_string_split[1].strip()
                if self.use_agent_utterance:
                    agent_utt = src_string_split[3].strip() 
                user_utt = src_string_split[5].strip()
            else:
                raise ValueError(f"Expected source string to have one or two turns, got {self.src_str}")
        else:
            user_utts = re.split("__User", self.src_str)
            last_user_utt = user_utts[-1]
            user_utt = re.sub(f"({PROGRAM_SEP}).*", "", last_user_utt) 
            user_utt = re.sub("(__User).*$", "", user_utt) 
            user_utt = user_utt.strip() 
            
        user_utt = user_utt.split(" ")  
        
        # prepend agent utterance 
        if agent_utt is not None:
            agent_utt = agent_utt.split(" ")
            user_utt = ["__Agent"] + agent_utt + ["__User"] + user_utt
        else:
            user_utt = ["__User"] + user_utt

        # prepend context utterance to user utterance 
        if prev_utt is not None:
            prev_utt = prev_utt.split(" ")
            user_utt = ["__User"] + prev_utt + user_utt

        if bert_tokenizer is not None:
            bert_tokenizer_ret = bert_tokenizer.tokenize(user_utt, True)
            src_token_ids = bert_tokenizer_ret["token_ids"]
            src_token_subword_index = bert_tokenizer_ret["token_recovery_matrix"]
        else:
            src_token_ids = None
            src_token_subword_index = None

        src_tokens = user_utt

        # TODO (elias): we could add some string normalization here to 
        # allow copies without punctuation, e.g. "Darby" in 
        # output can be copied from Darby in input and quotations 
        # can be deterministically added
        src_copy_vocab = SourceCopyVocabulary(src_tokens)
        src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
        src_copy_map = src_copy_vocab.get_copy_map(src_tokens)
        src_must_copy_tags = [0 for t in src_tokens]


        tgt_tokens_to_generate = tgt_tokens[:]
        # Replace all tokens that can be copied with OOV.
        for i, index in enumerate(src_copy_indices):
            if index != src_copy_vocab.token_to_idx[src_copy_vocab.unk_token]:
                tgt_tokens_to_generate[i] = DEFAULT_OOV_TOKEN

        if self.fxn_of_interest is not None and self.fxn_of_interest in tgt_tokens:
            contains_fxn = [1]

        else:
            contains_fxn = [0]


        return {
            "tgt_tokens" : tgt_tokens,
            "tgt_indices": tgt_indices,
            "tgt_copy_indices" : tgt_copy_indices,
            "tgt_copy_map" : tgt_copy_map,
            "tgt_tokens_to_generate": tgt_tokens_to_generate, 
            "src_tokens" : src_tokens,
            "src_token_ids" : src_token_ids,
            "src_token_subword_index" : src_token_subword_index,
            "src_must_copy_tags" : src_must_copy_tags,
            "src_copy_vocab" : src_copy_vocab,
            "src_copy_indices" : src_copy_indices,
            "src_copy_map" : src_copy_map,
            "src_str": self.src_str,
            "tgt_str": self.tgt_str,
            "contains_fxn": contains_fxn
        }

    @staticmethod
    def from_prediction(tgt_tokens):
        pred_lispress_str = " ".join(tgt_tokens)
        if "Func" in pred_lispress_str and "(" not in pred_lispress_str:
            # dealing with synthetic data 
            return pred_lispress_str

        # try a round trip to make sure output is valid 
        try:
            pred_lispress_str = render_compact(parse_lispress(pred_lispress_str))
        except: 
            pred_lispress_str = "(Error)"

        return pred_lispress_str