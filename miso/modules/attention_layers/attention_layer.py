# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from overrides import overrides
from allennlp.common import Registrable

from miso.modules.attention import Attention


class AttentionLayer(torch.nn.Module, Registrable):
    """
    An ``AttentionLayer'' takes three inputs: query, key and value.
    Attention weights are computed by an attention function over query and key.
    """
    def __init__(self,
                 attention: Attention):
        super().__init__()
        self.attention = attention

    @overrides
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """
        :param query: [batch_size, query_length, query_vector_dim].
        :param key: [batch_size, key_length, key_vector_dim].
        :param value: [batch_size, value_length, value_vector_dim].
        :return: [batch_size, query_length, value_vector_dim].
        """
        # [batch_size, query_length, key_length]
        attention_weights = self.attention(query, key)
        return torch.bmm(attention_weights, value)
