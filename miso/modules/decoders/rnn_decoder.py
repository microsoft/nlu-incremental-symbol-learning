# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Dict, Optional

from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.common.registrable import Registrable
from allennlp.modules import InputVariationalDropout

from miso.modules.stacked_lstm import StackedLstm
from miso.modules.attention_layers import AttentionLayer
from miso.modules.decoders.decoder import MisoDecoder

import logging 
logger = logging.getLogger(__name__) 


@MisoDecoder.register("rnn_decoder") 
class RNNDecoder(MisoDecoder):
    def __init__(self,
                 rnn_cell: StackedLstm,
                 source_attention_layer: AttentionLayer,
                 target_attention_layer: AttentionLayer = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.rnn_cell = rnn_cell
        self.source_attention_layer = source_attention_layer
        self.target_attention_layer = target_attention_layer
        self.dropout = InputVariationalDropout(dropout)
        self.use_coverage = source_attention_layer.attention.use_coverage
        self.hidden_vector_dim = self.rnn_cell.hidden_size

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                source_memory_bank: torch.Tensor,
                source_mask: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor]] = None,
                input_feed: Optional[torch.Tensor] = None) -> Dict:
        """
        Given a sequence of inputs, run Teacher Forcing RNN.
        :param inputs: [batch_size, input_seq_length, input_vector_dim].
        :param source_memory_bank: [batch_size, source_seq_length, source_vector_dim].
        :param source_mask: [batch_size, source_seq_length].
        :param hidden_state: a tuple of (LSTM state, LSTM memory) in shape [num_layers, batch_size, hidden_vector_dim].
        :param input_feed: [batch_size, 1, hidden_vector_dim].
        """
        # Output.
        attentional_tensors = []
        rnn_outputs = []
        source_attention_weights = []
        target_attention_weights = []
        coverage_history = []

        # Initialization.
        batch_size, input_seq_length, _ = inputs.size()
        _, source_seq_length, _ = source_memory_bank.size()
        if input_feed is None:
            input_feed = inputs.new_zeros(size=(batch_size, 1, self.hidden_vector_dim))
        if self.use_coverage:
            coverage = inputs.new_zeros(size=(batch_size, 1, source_seq_length))
        else:
            coverage = None
        coverage_history.append(coverage)

        # Step-by-step decoding.
        for step_i, one_step_input in enumerate(inputs.split(1, dim=1)):
            target_memory_bank = torch.cat(attentional_tensors, 1) if len(attentional_tensors) else None

            one_step_output = self.one_step_forward(
                input_tensor=one_step_input,
                source_memory_bank=source_memory_bank,
                source_mask=source_mask,
                target_memory_bank=target_memory_bank,
                decoding_step=step_i,
                total_decoding_steps=input_seq_length,
                input_feed=input_feed,
                hidden_state=hidden_state,
                coverage=coverage
            )
            input_feed = one_step_output["attentional_tensor"]
            hidden_state = one_step_output["hidden_state"]
            coverage = one_step_output["coverage"]

            attentional_tensors.append(one_step_output["attentional_tensor"])
            rnn_outputs.append(one_step_output["rnn_output"])
            source_attention_weights.append(one_step_output["source_attention_weights"])
            target_attention_weights.append(one_step_output["target_attention_weights"])
            coverage_history.append(coverage)

        # [batch_size, input_seq_length, vector_dim]
        attentional_tensors = torch.cat(attentional_tensors, 1)
        rnn_outputs = torch.cat(rnn_outputs, 1)
        # [batch_size, input_seq_length, source_seq_length]
        source_attention_weights = torch.cat(source_attention_weights, 1)
        # [batch_size, input_seq_length, target_seq_length]
        target_attention_weights = torch.cat(target_attention_weights, 1)
        # [batch_size, input_seq_length, source_seq_length] or None
        if self.use_coverage:
            coverage_history = torch.cat(coverage_history[:-1], 1)  # Exclude the last one.
        else:
            coverage_history = None

        return dict(
            attentional_tensors=attentional_tensors,
            rnn_outputs=rnn_outputs,
            source_attention_weights=source_attention_weights,
            target_attention_weights=target_attention_weights,
            coverage_history=coverage_history
        )

    def one_step_forward(self,
                         input_tensor: torch.Tensor,
                         source_memory_bank: torch.Tensor,
                         source_mask: torch.Tensor,
                         target_memory_bank: torch.Tensor = None,
                         decoding_step: int = 0,
                         total_decoding_steps: int = 0,
                         input_feed: Optional[torch.Tensor] = None,
                         hidden_state: Optional[Tuple[torch.Tensor]] = None,
                         coverage: Optional[torch.Tensor] = None) -> Dict:
        """
        Run a single step decoding.
        :param input_tensor: [batch_size, 1, input_vector_dim].
        :param source_memory_bank: [batch_size, source_seq_length, source_vector_dim].
        :param source_mask: [batch_size, source_seq_length].
        :param target_memory_bank: [batch_size, target_seq_length, target_vector_dim].
        :param decoding_step: index of the current decoding step.
        :param total_decoding_steps: the total number of decoding steps.
        :param input_feed: [batch_size, 1, hidden_vector_dim].
        :param hidden_state: a tuple of (LSTM state, LSTM memory) in shape [num_layers, batch_size, hidden_vector_dim].
        :param coverage: [batch_size, 1, source_seq_length].
        :return:
        """

        batch_size, source_seq_length, _ = source_memory_bank.size()
        if input_feed is None:
            input_feed = input_tensor.new_zeros(size=(batch_size, 1, self.hidden_vector_dim))
        if self.use_coverage and coverage is None:
            coverage = input_tensor.new_zeros(size=(batch_size, 1, source_seq_length))
        # RNN.
        concat_input = torch.cat([input_tensor, input_feed], 2)
        packed_input = pack_padded_sequence(concat_input, [1] * batch_size, batch_first=True)
        packed_output, hidden_state = self.rnn_cell(packed_input, hidden_state)
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)
    

        # source-side attention.
        source_attention_output = self.source_attention_layer(
            rnn_output, source_memory_bank, source_mask, coverage
        )
        attentional_tensor = self.dropout(source_attention_output["attentional"])
        source_attention_weights = source_attention_output["attention_weights"]
        coverage = source_attention_output["coverage"]

        # target-side attention.
        target_attention_weights = self._compute_target_attention(
            attentional_tensor, target_memory_bank, decoding_step, total_decoding_steps
        )

        return dict(
            attentional_tensor=attentional_tensor,
            rnn_output=rnn_output,
            source_attention_weights=source_attention_weights,
            target_attention_weights=target_attention_weights,
            hidden_state=hidden_state,
            coverage=coverage
        )

    def _compute_target_attention(self,
                                  query: torch.Tensor,
                                  key: torch.Tensor,
                                  decoding_step: int,
                                  total_decoding_steps: int) -> torch.Tensor:
        """
        Compute the target-side attention, and return a fixed length tensor
        representing attention weights for the current decoding step.
        :param query: [batch_size, 1, query_vector_dim].
        :param key: None or [batch_size, key_seq_length, key_vector_dim].
        :param decoding_step: index of the current decoding step.
        :param total_decoding_steps: the total number of decoding steps.
        :return: [batch_size, 1, total_decoding_steps].
        """
        if key is not None:
            attention_weights = self.target_attention_layer(query, key)["attention_weights"]
            if total_decoding_steps != 1:
                attention_weights = F.pad(attention_weights, [0, total_decoding_steps - decoding_step], "constant", 0)
        else:
            batch_size = query.size(0)
            attention_weights = query.new_zeros((batch_size, 1, total_decoding_steps))
        return attention_weights
