# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import torch
from overrides import overrides
from allennlp.common.registrable import Registrable


class Attention(torch.nn.Module, Registrable):
    """
    This is the base class of Attention used in Miso.
    We don't use the base class from AllenNLP because it requires us to override ``forward''
    and to implement ``_forward_internal''.
    """
    def __init__(self) -> None:
        super().__init__()

    @overrides
    def forward(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError
