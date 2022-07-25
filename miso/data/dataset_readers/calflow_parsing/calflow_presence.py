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


class CalFlowPresence:
    def __init__(self, 
                src_str: str,
                tgt_str: str,
                all_fxns: List[str], 
                use_agent_utterance: bool = False, 
                use_context: bool = False,
                fxn_of_interest: str = None):
        #self.program = self.tgt_str_to_program(tgt_str)
        self.src_str = src_str.strip() 
        self.tgt_str = tgt_str
        self.use_agent_utterance = use_agent_utterance
        self.use_context = use_context
        self.fxn_of_interest = fxn_of_interest
        self.all_fxns = all_fxns

    def get_list_data(self, bert_tokenizer): 
        """
        Converts SMCalFlow graph into set of indicators over functions 
        """
        tgt_tokens = self.tgt_str.strip().split(" ")

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

        src_tokens = user_utt

        contains_each = [0 for i in range(len(self.all_fxns))]

        for i in range(len(self.all_fxns)):
            fxn = self.all_fxns[i]
            if fxn in tgt_tokens:
                contains_each[i] = 1
            


        if self.fxn_of_interest is not None and self.fxn_of_interest in tgt_tokens:
            contains_fxn = [1]

        else:
            contains_fxn = [0]
            
        bert_tokenizer_ret = bert_tokenizer.tokenize(src_tokens, True)
        src_token_ids = bert_tokenizer_ret["token_ids"]

        return {
            "src_tokens": src_tokens,
            "src_str": self.src_str,
            "tgt_str": self.tgt_str,
            "src_token_ids": src_token_ids,
            "contains_each": contains_each,
            "contains_fxn": contains_fxn
        }
