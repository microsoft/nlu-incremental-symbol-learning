# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from allennlp.common.registrable import Registrable

class MisoDecoder(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__() 

    def forward(self, inputs): 
        pass 
