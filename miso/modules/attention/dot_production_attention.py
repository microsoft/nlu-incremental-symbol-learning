# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from overrides import overrides

from .attention import Attention


@Attention.register("dot_product")
class DotProductAttention(Attention):

    def __init__(self,
                 decoder_hidden_size: int,
                 encoder_hidden_size: int,
                 share_linear: bool =True) -> None:
        super(DotProductAttention, self).__init__()
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.linear_layer = torch.nn.Linear(decoder_hidden_size, encoder_hidden_size, bias=False)
        self.share_linear = share_linear

    @overrides
    def forward(self,
                decoder_input: torch.Tensor,
                encoder_input: torch.Tensor,
                encoder_mask: torch.Tensor = None,
                coverage: bool = None) -> torch.Tensor:
        """
        :param decoder_input:  [batch, decoder_seq_length, decoder_hidden_size]
        :param encoder_input:  [batch, encoder_seq_length, encoder_hidden_size]
        :return:  [batch, decoder_seq_length, encoder_seq_length]
        """
        decoder_input = self.linear_layer(decoder_input)
        if self.share_linear:
            encoder_input = self.linear_layer(encoder_input)

        encoder_input = encoder_input.transpose(1, 2)
        return torch.bmm(decoder_input, encoder_input)
