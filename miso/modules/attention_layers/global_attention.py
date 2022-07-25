# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

from overrides import overrides
import torch
from allennlp.nn.util import masked_softmax

from miso.modules.attention import Attention
from .attention_layer import AttentionLayer


@AttentionLayer.register("global")
class GlobalAttention(AttentionLayer):

    def __init__(self,
                 query_vector_dim: int,
                 key_vector_dim: int,
                 output_vector_dim: int,
                 attention: Attention) -> None:
        super().__init__(attention)
        self.query_vector_dim = query_vector_dim
        self.key_vector_dim = key_vector_dim
        self.output_layer = torch.nn.Linear(
            query_vector_dim + key_vector_dim,
            output_vector_dim,
            bias=True
        )

    @overrides
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                mask: torch.Tensor = None,
                coverage: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        :param query: [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param mask: fill with pad with 0, [batch_size, key_length]
        :param coverage: [batch_size, query_length, key_length]
        """
        # Output: [batch_size, query_length, key_length]
        if coverage is not None:
            attention_weights = self.attention(query, key, coverage)
        else:
            attention_weights = self.attention(query, key)

        # Normalize: [batch_size, query_length, key_length]
        attention_weights = masked_softmax(attention_weights, mask, 2, memory_efficient=True)

        # [batch_size, query_length, key_vector_dim]
        context = torch.bmm(attention_weights, key)

        # [batch_size, query_length, output_vector_dim]
        attentional = torch.tanh(self.output_layer(torch.cat([context, query], 2)))

        if coverage is not None:
            coverage = coverage + attention_weights

        return {
            "attentional": attentional,
            "attention_weights": attention_weights,
            "coverage": coverage
        }
