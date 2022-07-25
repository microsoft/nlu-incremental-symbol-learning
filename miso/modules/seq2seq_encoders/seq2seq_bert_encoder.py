# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import logging
import pdb 

from allennlp.common import Registrable
from transformers import BertModel, XLMRobertaModel, RobertaModel, AlbertModel

logger = logging.getLogger(__name__) 

class BaseBertWrapper(Registrable, torch.nn.Module):

    def __init__(self, config: str, 
                       model_class = BertModel) -> None:
        super().__init__()
        self.bert_model = model_class.from_pretrained(config).eval()

@BaseBertWrapper.register("seq2seq_bert_encoder")
class Seq2SeqBertEncoder(BaseBertWrapper):

    def __init__(self, config: str) -> None:
        super(Seq2SeqBertEncoder, self).__init__(config, BertModel) 

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                output_all_encoded_layers: bool = True,
                token_recovery_matrix: torch.LongTensor = None) -> torch.Tensor:
        """
        :param input_ids: same as it in BertModel
        :param token_type_ids: same as it in BertModel
        :param attention_mask: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_recovery_matrix: [batch_size, num_tokens, num_subwords]
        """
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]
        # with torch.no_grad():
        encoded_layers, __ = self.bert_model(
            input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        #encoded_layers = output['last_hidden_state']
        if token_recovery_matrix is None:
            return encoded_layers
        else:
            return average_pooling(encoded_layers, token_recovery_matrix)

@BaseBertWrapper.register("seq2seq_xlmr_encoder")
class Seq2SeqXLMRobertaEncoder(BaseBertWrapper):

    def __init__(self, config, use_bert_all_layers=False):
        super(Seq2SeqXLMRobertaEncoder, self).__init__(config, XLMRobertaModel)

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                output_all_encoded_layers: bool = True,
                token_recovery_matrix: torch.LongTensor = None) -> torch.Tensor:
        """
        :param input_ids: same as it in BertModel
        :param token_type_ids: same as it in BertModel
        :param attention_mask: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_recovery_matrix: [batch_size, num_tokens, num_subwords]
        """
        max_len = 512
        # with torch.no_grad(): 
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]
        encoded_layers, __ = self.bert_model(
            input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        #encoded_layers = output['last_hidden_state']

        if token_recovery_matrix is None:
            return encoded_layers
        else:
            #encoded_layers = encoded_layers[:, 0:max_len-10, :]
            #token_recovery_matrix = token_recovery_matrix[:,0:max_len-10,:]
            return average_pooling(encoded_layers, token_recovery_matrix)


def average_pooling(encoded_layers: torch.FloatTensor,
                    token_subword_index: torch.LongTensor) -> torch.Tensor:
    batch_size, num_tokens, num_subwords = token_subword_index.size()
    batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
    token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
    _, num_total_subwords, hidden_size = encoded_layers.size()
    expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
        batch_size, num_tokens, num_total_subwords, hidden_size)
    # [batch_size, num_tokens, num_subwords, hidden_size]
    token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
    subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
        batch_size, num_tokens, num_subwords, hidden_size)
    token_reprs.masked_fill_(subword_pad_mask, 0)
    # [batch_size, num_tokens, hidden_size]
    sum_token_reprs = torch.sum(token_reprs, dim=2)
    # [batch_size, num_tokens]
    num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
    pad_mask = num_valid_subwords.eq(0).long()
    # Add ones to arrays where there is no valid subword.
    divisor = (num_valid_subwords + pad_mask).unsqueeze(2).type_as(sum_token_reprs)
    # [batch_size, num_tokens, hidden_size]
    avg_token_reprs = sum_token_reprs / divisor
    return avg_token_reprs


def max_pooling(encoded_layers: torch.FloatTensor,
                token_subword_index: torch.LongTensor) -> torch.Tensor:
    batch_size, num_tokens, num_subwords = token_subword_index.size()
    batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
    token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
    _, num_total_subwords, hidden_size = encoded_layers.size()
    expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
        batch_size, num_tokens, num_total_subwords, hidden_size)
    # [batch_size, num_tokens, num_subwords, hidden_size]
    token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
    subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
        batch_size, num_tokens, num_subwords, hidden_size)
    token_reprs.masked_fill_(subword_pad_mask, -float('inf'))
    # [batch_size, num_tokens, hidden_size]
    max_token_reprs, _ = torch.max(token_reprs, dim=2)
    # [batch_size, num_tokens]
    num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
    pad_mask = num_valid_subwords.eq(0).unsqueeze(2).expand(
        batch_size, num_tokens, hidden_size)
    max_token_reprs.masked_fill(pad_mask, 0)
    return max_token_reprs


@BaseBertWrapper.register("seq2seq_roberta_encoder")
class Seq2SeqRobertaEncoder(BaseBertWrapper):

    def __init__(self, config, use_bert_all_layers=False):
        super(Seq2SeqRobertaEncoder, self).__init__(config, RobertaModel)

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                output_all_encoded_layers: bool = True,
                token_recovery_matrix: torch.LongTensor = None) -> torch.Tensor:
        """
        :param input_ids: same as it in BertModel
        :param token_type_ids: same as it in BertModel
        :param attention_mask: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_recovery_matrix: [batch_size, num_tokens, num_subwords]
        """
        max_len = 512
        # with torch.no_grad(): 
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]
        encoded_layers, __ = self.bert_model(
            input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        #encoded_layers = output['last_hidden_state']

        if token_recovery_matrix is None:
            return encoded_layers
        else:
            #encoded_layers = encoded_layers[:, 0:max_len-10, :]
            #token_recovery_matrix = token_recovery_matrix[:,0:max_len-10,:]
            return average_pooling(encoded_layers, token_recovery_matrix)


@BaseBertWrapper.register("seq2seq_albert_encoder")
class Seq2SeqAlbertEncoder(BaseBertWrapper):

    def __init__(self, config, use_bert_all_layers=False):
        super(Seq2SeqAlbertEncoder, self).__init__(config, AlbertModel)

    def forward(self,
                input_ids: torch.LongTensor,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                output_all_encoded_layers: bool = True,
                token_recovery_matrix: torch.LongTensor = None) -> torch.Tensor:
        """
        :param input_ids: same as it in BertModel
        :param token_type_ids: same as it in BertModel
        :param attention_mask: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_recovery_matrix: [batch_size, num_tokens, num_subwords]
        """
        max_len = 512
        # with torch.no_grad(): 
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]
        encoded_layers, __ = self.bert_model(
            input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        #encoded_layers = output['last_hidden_state']

        if token_recovery_matrix is None:
            return encoded_layers
        else:
            #encoded_layers = encoded_layers[:, 0:max_len-10, :]
            #token_recovery_matrix = token_recovery_matrix[:,0:max_len-10,:]
            return average_pooling(encoded_layers, token_recovery_matrix)

