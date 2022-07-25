# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any 
import logging
import pdb
from collections import OrderedDict

from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder 
from overrides import overrides
import torch

from allennlp.data import Token, Instance, Vocabulary
from allennlp.models import Model
from torch.nn.modules import activation
from torch.nn.modules.activation import ReLU
from transformers import PreTrainedTokenizer, PreTrainedModel, BertModel, BertTokenizer
from allennlp.training.metrics import BooleanAccuracy

from miso.modules.decoders.binary_classfier import BaseBinaryClassifier


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("calflow_mlp")
class CalFlowMLP(Model):

    def __init__(self,
                 # source-side
                 vocab: Vocabulary, 
                 encoder_token_embedder: TextFieldEmbedder,
                 input_dim: int = 16,
                 hidden_dim: int = 128,
                 output_dim: int = 2,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 fxn_of_interest: str = None) -> None:
        super().__init__(vocab)
        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self._encoder_token_embedder = encoder_token_embedder
        self.fxn_of_interest = fxn_of_interest
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = torch.nn.Dropout(p=dropout)
        self.input_to_hidden = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = [torch.nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers-1)]
        self.hidden_to_output = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        modules = [self.input_to_hidden, self.activation, self.dropout] 
        for hl in self.hidden_layers:
            modules.append(hl)
            modules.append(self.activation)
            modules.append(self.dropout)
        modules += [self.hidden_to_output]
        self.network = torch.nn.Sequential(*modules)
        self.accuracy_metric = BooleanAccuracy()

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.accuracy_metric.get_metric(reset) 
        metrics = OrderedDict(acc=accuracy)
        return metrics

    def _training_forward(self, inputs):
        tokens = inputs["source_tokens"]
        encoder_inputs = [self._encoder_token_embedder(tokens)][0]
        encoder_outputs = self.network(encoder_inputs).squeeze(1)
        gold = inputs["target_labels"].reshape(-1).long()
        loss = self.loss_fxn(encoder_outputs, gold) 
        self.accuracy_metric(torch.argmax(encoder_outputs, dim=1), gold)
        return {"loss": loss,
                "output": encoder_outputs}

    def _test_forward(self, inputs):
        with torch.no_grad():
            tokens = inputs["source_tokens"]
            encoder_inputs = [self._encoder_token_embedder(tokens)][0]
            encoder_outputs = self.network(encoder_inputs).squeeze(1)
            gold = inputs["target_labels"].reshape(-1).long()
            self.accuracy_metric(torch.argmax(encoder_outputs, dim=1), gold)
            return {"output": encoder_outputs,
                    "src_str": inputs['src_str']}


    @overrides
    def forward(self, **inputs: Dict) -> Dict:
        if self.training:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs)