# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any 
import logging

import pdb 
from overrides import overrides
import torch

from allennlp.data import Token, Instance, Vocabulary
from allennlp.models import Model
from transformers import PreTrainedTokenizer, PreTrainedModel, BertModel, BertTokenizer

from miso.modules.decoders.binary_classfier import BaseBinaryClassifier


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("calflow_presence_parser")
class PresenceCalFlowParser(Model):

    def __init__(self,
                 # source-side
                 vocab: Vocabulary, 
                 bert_name: str = "bert-base-cased",
                 output_module: BaseBinaryClassifier = None,
                 fxn_of_interest: str = None) -> None:
        super().__init__(vocab)
        self._bert_model = BertModel.from_pretrained(bert_name)
        self._bert_model.eval()
        self._bert_model.to('cuda') 
        self._output_module = output_module
        self.loss_fxn = torch.nn.CrossEntropyLoss()
        self.fxn_of_interest = fxn_of_interest

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass 

    def _training_forward(self, inputs):
        # freeze bert 
        with torch.no_grad():
            encoded = self._bert_model(inputs['src_token_ids'].long()) 
        # take [CLS] token
        output_array = self._output_module(encoded[0][:,0,:])

        output_array = output_array.reshape(-1, 2)
        gold = inputs["contains_each"].reshape(-1).long()

        loss = self.loss_fxn(output_array, gold) 
        return {"loss": loss,
                "output": output_array}


    def _test_forward(self, inputs):
        with torch.no_grad():
            tokenized = self._bert_tokenizer(inputs['src_tokens'].long())
            # take [CLS] token
            encoded = self._bert_model(tokenized)
        output_array = self._output_module(encoded[0][:,0,:])

        sigged = torch.sigmoid(output_array)
        return {"output": torch.argmax(sigged, dim=1)}


    @overrides
    def forward(self, **inputs: Dict) -> Dict:
        if self.training:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs)