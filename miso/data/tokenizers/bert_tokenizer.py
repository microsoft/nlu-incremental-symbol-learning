# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides

import numpy as np
from transformers import PreTrainedTokenizer, BertTokenizer, XLMRobertaTokenizer, AutoTokenizer, AutoConfig, RobertaTokenizer, AlbertTokenizer
from allennlp.common import Params
from allennlp.data.tokenizers import Tokenizer
from allennlp.common.registrable import Registrable

def tokenize_helper(tokenizer, tokens, split=False):
    assert isinstance(tokenizer, PreTrainedTokenizer)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    if not split:
        split_tokens = tokens
        gather_indexes = None
    else:
        split_tokens, _gather_indexes = [], []
        for token in tokens:
            indexes = []
            for i, sub_token in enumerate(tokenizer._tokenize(token)):
                indexes.append(len(split_tokens))
                split_tokens.append(sub_token)
            _gather_indexes.append(indexes)

        _gather_indexes = _gather_indexes[1:-1]
        max_index_list_len = max(len(indexes) for indexes in _gather_indexes)
        gather_indexes = np.zeros((len(_gather_indexes), max_index_list_len))
        for i, indexes in enumerate(_gather_indexes):
            for j, index in enumerate(indexes):
                gather_indexes[i, j] = index

    token_ids = np.array(tokenizer.convert_tokens_to_ids(split_tokens))
    return {"token_ids": token_ids, 
            "token_recovery_matrix": gather_indexes}

class MisoTokenizer(Registrable):
    def __init__(self):
        pass



@MisoTokenizer.register("pretrained_transformer_for_amr") 
class AMRBertTokenizer(BertTokenizer):
    def __init__(self, model_name: str,
                args: None, # extra args to make backwards-compatible
                kwargs: None):

        # Hacky fix to get to play nice with registering and pretrained 
        tok = BertTokenizer.from_pretrained(model_name)
        self.__dict__ = tok.__dict__

    @overrides
    def tokenize(self, tokens, split=False):
        return tokenize_helper(self, tokens, split=split)

@MisoTokenizer.register("pretrained_xlmr") 
class AMRXLMRobertaTokenizer(XLMRobertaTokenizer):
    def __init__(self, model_name: str): 
        self.model_name = model_name

    def __init__(self, model_name: str): 
        # Hacky fix to get to play nice with registering and pretrained 
        tok = XLMRobertaTokenizer.from_pretrained(model_name)
        self.__dict__ = tok.__dict__

    @overrides
    def tokenize(self, tokens, split=False):
        return tokenize_helper(self, tokens, split=split)

@MisoTokenizer.register("pretrained_roberta") 
class AMRRobertaTokenizer(RobertaTokenizer):
    def __init__(self, model_name: str): 
        self.model_name = model_name

    def __init__(self, model_name: str): 
        # Hacky fix to get to play nice with registering and pretrained 
        tok = RobertaTokenizer.from_pretrained(model_name)
        self.__dict__ = tok.__dict__

    @overrides
    def tokenize(self, tokens, split=False):
        return tokenize_helper(self, tokens, split=split)


@MisoTokenizer.register("pretrained_albert") 
class AMRAlbertTokenizer(AlbertTokenizer):
    def __init__(self, model_name: str): 
        self.model_name = model_name

    def __init__(self, model_name: str): 
        # Hacky fix to get to play nice with registering and pretrained 
        tok = AlbertTokenizer.from_pretrained(model_name)
        self.__dict__ = tok.__dict__

    @overrides
    def tokenize(self, tokens, split=False):
        return tokenize_helper(self, tokens, split=split)

