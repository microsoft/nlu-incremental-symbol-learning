# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import json
import sys
import logging
from collections import defaultdict, Counter, namedtuple
from typing import List, Dict
from overrides import overrides

from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
import spacy 

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class UDGraph:
    def __init__(self, conllu_dict): 
        self.conllu_dict = conllu_dict

    def get_list_data(self, bert_tokenizer=None): 
        syn_tokens, syn_head_indices, syn_head_tags = [], [], []
        src_tokens, src_token_ids, src_pos_tags, src_token_subword_index = [], [], [], []

        #colnames = ["ID", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
        for row in self.conllu_dict: 
            # src stuff 
            src_tokens.append(row["form"]) 
            src_pos_tags.append(row["upos"])
            # syn stuff 
            syn_tokens.append(row["form"])
            syn_head_indices.append(row["head"])
            syn_head_tags.append(row["deprel"])

        # Source Info
        src_token_ids = None
        src_token_subword_index = None

        if len(src_pos_tags) == 0:
            # happens when predicting from just a sentence
            # use spacy to get a POS tag sequence 
            doc = nlp(" ".join(src_tokens).strip())
            src_pos_tags = [token.pos_ for token in doc]

        if bert_tokenizer is not None:
            bert_tokenizer_ret = bert_tokenizer.tokenize(src_tokens, True)
            src_token_ids = bert_tokenizer_ret["token_ids"]
            src_token_subword_index = bert_tokenizer_ret["token_recovery_matrix"]

        true_conllu_dict = self.conllu_dict

        syn_node_mask = np.array([1] * len(syn_tokens), dtype='uint8')
        syn_edge_mask = np.ones((len(syn_tokens), len(syn_tokens)), dtype='uint8')

        return {
            "syn_tokens": syn_tokens, 
            "syn_head_indices": syn_head_indices,
            "syn_head_tags": syn_head_tags,
            "syn_node_mask": syn_node_mask,
            "syn_edge_mask": syn_edge_mask,
            "src_tokens" : src_tokens,
            "src_pos_tags": src_pos_tags,
            "src_token_ids" : src_token_ids,
            "src_token_subword_index" : src_token_subword_index,
            "true_conllu_dict": true_conllu_dict,
        }
