# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Dict, Optional
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_log_softmax
from miso.modules.attention import Attention
from miso.modules.parsers import DeepTreeParser

class DecompTreeParser(DeepTreeParser):

    @overrides
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                edge_head_mask: torch.ByteTensor = None,
                gold_edge_heads: torch.Tensor = None
                ) -> Dict:
        """
        :param query: [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head_mask: [batch_size, query_length, key_length]
                        1 indicates a valid position; otherwise, 0.
        :param gold_edge_heads: None or [batch_size, query_length].
                        head indices start from 1.
        :return:
            edge_heads: [batch_size, query_length].
            edge_types: [batch_size, query_length].
            edge_head_ll: [batch_size, query_length, key_length + 1(sentinel)].
            edge_type_ll: [batch_size, query_length, num_labels] (based on gold_edge_head) or None.
        """
        if gold_edge_heads is not None:
            gold_edge_heads[gold_edge_heads == -1] = 0

        key, edge_head_mask = self._add_sentinel(query, key, edge_head_mask)
        edge_head_query, edge_head_key, edge_type_query, edge_type_key = self._mlp(query, key)
        # [batch_size, query_length, key_length + 1]
        edge_head_score = self._get_edge_head_score(edge_head_query, edge_head_key)
        edge_heads, edge_types = self._greedy_search(
            edge_type_query, edge_type_key, edge_head_score, edge_head_mask
        )

        if gold_edge_heads is None:
            gold_edge_heads = edge_heads
        # [batch_size, query_length, num_labels]
        edge_type_score = self._get_edge_type_score(edge_type_query, edge_type_key, edge_heads)

        return dict(
            # Note: head indices start from 1.
            edge_heads=edge_heads,
            edge_types=edge_types,
            # Log-Likelihood.
            edge_head_ll=masked_log_softmax(edge_head_score, edge_head_mask, dim=2),
            edge_type_ll=masked_log_softmax(edge_type_score, None, dim=2),
            edge_head_query=edge_head_query,
            edge_head_key=edge_head_key,
            edge_type_query=edge_type_query,
            edge_type_key=edge_type_key
        )

