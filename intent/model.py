# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb 
import torch
from transformers import AutoModel

class Classifier(torch.nn.Module):
    def __init__(self,
                 bert_name: str, 
                 num_classes: int = 68):
        super(Classifier, self).__init__()
        self.bert_encoder = AutoModel.from_pretrained(bert_name)
        self.output_layer = torch.nn.Linear(768, num_classes)

    def forward(self, batch):
        encoded = self.bert_encoder(batch['input'])
        # get last encoded at [CLS]
        return self.output_layer(encoded[0][:,0,:])