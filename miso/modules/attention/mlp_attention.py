# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from overrides import overrides

from .attention import Attention


@Attention.register("mlp")
class MLPAttention(Attention):

    def __init__(self,
                 query_vector_dim: int,
                 key_vector_dim: int,
                 hidden_vector_dim: int,
                 use_coverage: bool = False) -> None:
        super().__init__()
        self.query_linear = torch.nn.Linear(query_vector_dim, hidden_vector_dim, bias=True)
        self.key_linear = torch.nn.Linear(key_vector_dim, hidden_vector_dim, bias=False)
        self.output_linear = torch.nn.Linear(hidden_vector_dim, 1, bias=False)
        if use_coverage:
            self.coverage_linear = torch.nn.Linear(1, hidden_vector_dim, bias=False)
        self._hidden_vector_dim = hidden_vector_dim
        self._use_coverage = use_coverage

    @property
    def hidden_vector_dim(self) -> int:
        return self._hidden_vector_dim

    @property
    def use_coverage(self) -> bool:
        return self._use_coverage

    @overrides
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                coverage: torch.Tensor = None) -> torch.Tensor:
        """
        :param query:  [batch_size, query_seq_length, query_vector_dim].
        :param key:  [batch_size, key_seq_length, key_vector_dim].
        :param coverage: [batch_size, key_seq_length]
        :return:  [batch_size, query_seq_length, key_seq_length]
        """
        batch_size, query_seq_length, query_vector_dim = query.size()
        batch_size, key_seq_length, key_vector_dim = key.size()

        query_linear_output = self.query_linear(query).unsqueeze(2).expand(
            batch_size, query_seq_length, key_seq_length, self._hidden_vector_dim
        )
        key_linear_output = self.key_linear(key).unsqueeze(1).expand(
            batch_size, query_seq_length, key_seq_length, self._hidden_vector_dim)

        activation_input = query_linear_output + key_linear_output

        if self._use_coverage:
            coverage_linear_output = self.coverage_linear(coverage.view(batch_size, 1, key_seq_length, 1))
            coverage_linear_output = coverage_linear_output.expand(
                batch_size, query_seq_length, key_seq_length, self._hidden_vector_dim
            )
            activation_input = activation_input + coverage_linear_output

        attention_weights = self.output_linear(torch.tanh(activation_input)).squeeze(3)

        return attention_weights
