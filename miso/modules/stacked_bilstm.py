# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder


Seq2SeqEncoder.register("miso_stacked_bilstm")
class MisoStackedBidirectionalLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "stacked_bidirectional_lstm".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        recurrent_dropout_probability: float = 0.0,
        layer_dropout_probability: float = 0.0,
        use_highway: bool = True,
        stateful: bool = False,
    ) -> None:
        module = MisoStackedBidirectionalLstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            recurrent_dropout_probability=recurrent_dropout_probability,
            layer_dropout_probability=layer_dropout_probability,
            use_highway=use_highway,
        )
        super().__init__(module=module, stateful=stateful)

    #def get_final_states(self) -> RnnStateStorage:
    #    return self._states

class MisoStackedBidirectionalLstm(StackedBidirectionalLstm):
    """
    The `StackedBidirentionalLstm` in Allennlp returns the final state tuple
    in the shape (num_layers * 2, batch_size, hidden_size) which is incompatible with
    the initial hidden state for the `RNNDecoder` in Miso.
    Here we overwrite the forward method to return a compatible final state tuple.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self,  # pylint: disable=arguments-differ
                inputs: PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : Tuple[torch.Tensor, torch.Tensor], optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. Each tensor has shape (num_layers, batch_size, output_dimension * 2).

        Returns
        -------
        output_sequence : PackedSequence
            The encoded sequence of shape (batch_size, sequence_length, hidden_size * 2)
        final_states: torch.Tensor
            The per-layer final (state, memory) states of the LSTM, each with shape
            (num_layers, batch_size, hidden_size * 2).
        """
        if not initial_state:
            hidden_states = [None] * len(self.lstm_layers)
        elif initial_state[0].size()[0] != len(self.lstm_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                                     initial_state[1].split(1, 0)))

        output_sequence = inputs
        final_h = []
        final_c = []
        for i, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(i))
            backward_layer = getattr(self, 'backward_layer_{}'.format(i))
            # The state is duplicated to mirror the Pytorch API for LSTMs.
            forward_output, final_forward_state = forward_layer(output_sequence, state)
            backward_output, final_backward_state = backward_layer(output_sequence, state)

            forward_output, lengths = pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = pad_packed_sequence(backward_output, batch_first=True)

            output_sequence = torch.cat([forward_output, backward_output], -1)
            # Apply layer wise dropout on each output sequence apart from the
            # first (input) and last
            if i < (self.num_layers - 1):
                output_sequence = self.layer_dropout(output_sequence)
            output_sequence = pack_padded_sequence(output_sequence, lengths, batch_first=True)

            final_h.append(torch.cat([final_forward_state[0], final_backward_state[0]], -1))
            final_c.append(torch.cat([final_forward_state[1], final_backward_state[1]], -1))

        final_state_tuple = (torch.cat(final_h, 0), torch.cat(final_c, 0))
        return output_sequence, final_state_tuple
