# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.encoder_base import RnnStateStorage

from miso.modules.stacked_bilstm import MisoStackedBidirectionalLstm

from .seq2seq_bert_encoder import Seq2SeqBertEncoder, BaseBertWrapper, Seq2SeqRobertaEncoder


class _PytorchSeq2SeqWrapper(PytorchSeq2SeqWrapper):
    """
    Inherit ``PytorchSeq2SeqWrapper'' to expose the final states.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_final_states(self) -> RnnStateStorage:
        return self._states


class _MisoSeq2SeqWrapper(_Seq2SeqWrapper):
    """
    It inherits ``_Seq2SeqWrapper'' from AllenNLP such that we have more flexibility
    of defining our own seq2seq encoders.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, **kwargs) -> PytorchSeq2SeqWrapper:
        return self.from_params(Params(kwargs))

    # Logic requires custom from_params
    def from_params(self, params: Params) -> _PytorchSeq2SeqWrapper:
        if not params.pop_bool('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        stateful = params.pop_bool('stateful', False)
        module = self._module_class(**params.as_dict(infer_type_and_cast=True))
        return _PytorchSeq2SeqWrapper(module, stateful=stateful)


# pylint: disable=protected-access
Seq2SeqEncoder.register("miso_stacked_bilstm")(_MisoSeq2SeqWrapper(MisoStackedBidirectionalLstm))
