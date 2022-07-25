# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import logging

from allennlp.common.registrable import Registrable
from torch.nn.modules import dropout

class BaseBinaryClassifier(torch.nn.Module, Registrable):
    def __init__(self,
                input_dim: int, 
                output_dim: int = 2):
        super(BaseBinaryClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, inputs):
        pass  

@BaseBinaryClassifier.register("shared")
class SharedBinaryClassifier(BaseBinaryClassifier):
    def __init__(self,
                input_dim: int, 
                output_dim: int,
                num_fxns: int,
                param_budget: int,
                dropout: float):
        super(SharedBinaryClassifier, self).__init__(input_dim, 
                                                    output_dim)
        self.num_fxns = num_fxns
        self.hidden_dim = param_budget
        self.input_layer = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.output_classfiers = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.output_dim) for __ in range(num_fxns)])
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.network = torch.nn.Sequential(self.input_layer, self.activation, self.dropout)

    def forward(self, input): 
        # input: batch, input_dim
        pre_output = self.network(input)
        # pre_output: batch, hidden_dim
        output = [classifier(pre_output).unsqueeze(1) for classifier in self.output_classfiers]
        # output: batch, num_fxns, output_dim
        output = torch.cat(output, dim = 1)
        return output 


@BaseBinaryClassifier.register("separate")
class SeparateBinaryClassifier(BaseBinaryClassifier):
    def __init__(self,
                input_dim: int, 
                output_dim: int,
                num_fxns: int,
                param_budget: int,
                dropout: float):
        super(SeparateBinaryClassifier, self).__init__(input_dim, 
                                                    output_dim)
        self.num_fxns = num_fxns
        self.hidden_dim = int(param_budget/num_fxns)
        self.input_layer = torch.nn.ModuleList([torch.nn.Linear(self.input_dim, self.hidden_dim) for __ in range(num_fxns)])
        self.output_classfiers = torch.nn.ModuleList([torch.nn.Linear(self.hidden_dim, self.output_dim) for __ in range(num_fxns)])
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.networks = torch.nn.ModuleList([torch.nn.Sequential(self.input_layer[i], self.activation, self.dropout, self.output_classifiers[i])for i in range(num_fxns)])

    def forward(self, input): 
        output = [network(input).unsqueeze(1) for network in self.networks]
        output = torch.cat(output, dim = 1)
        return output 

