# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from .attention import Attention


@Attention.register("biaffine")
class BiaffineAttention(Attention):
    """
    Adopted from NeuroNLP2:
        https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/nn/modules/attention.py

    Bi-Affine attention layer.
    """

    def __init__(self,
                 query_vector_dim: int,
                 key_vector_dim: int,
                 num_labels: int = 1,
                 use_bilinear: bool = True) -> None:
        super(BiaffineAttention, self).__init__()
        self.query_vector_dim = query_vector_dim
        self.key_vector_dim = key_vector_dim
        self.num_labels = num_labels
        self._use_bilinear = use_bilinear

        self.W_q = Parameter(torch.Tensor(num_labels, query_vector_dim))
        self.W_k = Parameter(torch.Tensor(num_labels, key_vector_dim))
        self.b = Parameter(torch.Tensor(num_labels, 1, 1))
        if use_bilinear:
            self.U = Parameter(torch.Tensor(num_labels, query_vector_dim, key_vector_dim))
        else:
            self.register_parameter('U', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.W_q)
        torch.nn.init.xavier_normal_(self.W_k)
        torch.nn.init.constant_(self.b, 0.)
        if self._use_bilinear:
            torch.nn.init.xavier_uniform_(self.U)

    @overrides
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                query_mask: torch.Tensor = None,
                key_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_length, query_vector_dim]
            key: [batch_size, key_length, key_vector_dim]
            query_mask: None or [batch_size, query_length]
            key_mask: None or [batch_size, key_length]
        Returns:
            the energy tensor with shape = [batch_size, num_labels, query_length, key_length]
        """
        batch_size, query_length, _ = query.size()
        _, key_length, _ = key.size()

        # Input: [num_labels, query_vector_dim] * [batch_size, query_vector_dim, query_length]
        # Output: [batch_size, num_labels, query_length, 1]
        query_linear_output = torch.matmul(self.W_q, query.transpose(1, 2)).unsqueeze(3)

        # Input: [num_labels, key_vector_dim] * [batch_size, key_vector_dim, key_length]
        # Output: [batch_size, num_labels, 1, key_length]
        key_linear_output = torch.matmul(self.W_k, key.transpose(1, 2)).unsqueeze(2)

        if self._use_bilinear:
            # Input: [batch_size, 1, query_length, query_vector_dim] * [num_labels, query_vector_dim, key_vector_dim]
            # Output: [batch_size, num_labels, query_length, key_vector_dim]
            bilinear_output = torch.matmul(query.unsqueeze(1), self.U)
            # Input: [batch_size, num_labels, query_length, key_vector_dim]*[batch_size, 1, key_vector_dim, key_length]
            # Output: [batch_size, num_labels, query_length, key_length]
            blinear_output = torch.matmul(bilinear_output, key.unsqueeze(1).transpose(2, 3))

            output = blinear_output + query_linear_output + key_linear_output + self.b
        else:
            output = query_linear_output + key_linear_output + self.b

        if query_mask is not None and key_mask is not None:
            output = output * query_mask.unsqueeze(1).unsqueeze(3) * key_mask.unsqueeze(1).unsqueeze(2)

        return output
