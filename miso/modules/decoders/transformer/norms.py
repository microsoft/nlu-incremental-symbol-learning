# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from overrides import overrides
from allennlp.common.registrable import Registrable

class Norm(torch.nn.Module, Registrable):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        pass 

@Norm.register("layer_norm") 
class LayerNorm(Norm):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(dim) 

    @overrides
    def forward(self, x):
        return self.norm(x) 

@Norm.register("scale_norm") 
class ScaleNorm(Norm):
    """ScaleNorm stolen from https://github.com/tnq177/transformers_without_tears/blob/master/layers.py"""
    def __init__(self, dim, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.tensor(dim**0.5, dtype=torch.float))
        self.eps = eps

    @overrides 
    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
