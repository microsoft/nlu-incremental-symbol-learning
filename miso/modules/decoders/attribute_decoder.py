# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import logging

from miso.metrics.continuous_metrics import ContinuousMetric
from miso.losses.loss import MSECrossEntropyLoss, Loss
from scipy.stats import pearsonr 

logger = logging.getLogger(__name__) 

np.set_printoptions(suppress=True)

class NodeAttributeDecoder(torch.nn.Module):
    def __init__(self,
                input_dim, 
                hidden_dim, 
                output_dim,
                n_layers,
                loss_multiplier = 10,
                loss_function = Loss,
                activation = torch.nn.ReLU(),
                share_networks = False,
                dropout = 0.20, 
                binary = False):
        super(NodeAttributeDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.loss_multiplier = loss_multiplier
        self.binary = binary 

        self.attr_loss_function = loss_function
        self.mask_loss_function = torch.nn.BCEWithLogitsLoss()

        self.dropout = torch.nn.Dropout(dropout) 

        self.n_layers = n_layers
        self.activation = activation
        

        attr_input_layer = torch.nn.Linear(input_dim, hidden_dim)
        attr_hidden_layers = [torch.nn.Linear(hidden_dim, hidden_dim) 
                            for i in range(n_layers-1)]

        boolean_input_layer = torch.nn.Linear(input_dim, hidden_dim)
        boolean_hidden_layers = [torch.nn.Linear(hidden_dim, hidden_dim) 
                            for i in range(n_layers-1)]
    
        attr_output_layer = torch.nn.Linear(hidden_dim, output_dim)
        boolean_output_layer = torch.nn.Linear(hidden_dim, output_dim)

        all_attr_layers = [attr_input_layer] + attr_hidden_layers 
        all_boolean_layers = [boolean_input_layer] + boolean_hidden_layers 
         
        attr_net =   []
        boolean_net =   []
        for l in all_attr_layers:
            attr_net.append(l)
            attr_net.append(self.activation)
            attr_net.append(self.dropout) 

        for l in all_boolean_layers:
            boolean_net.append(l)
            boolean_net.append(self.activation)
            boolean_net.append(self.dropout) 

        attr_net.append(attr_output_layer)
        boolean_net.append(boolean_output_layer)

        self.attribute_network = torch.nn.Sequential(*attr_net)

        if share_networks:
            self.boolean_network = self.attribute_network
        else:
            self.boolean_network = torch.nn.Sequential(*boolean_net)

        self.metrics = ContinuousMetric(prefix = "node")

    def forward(self, 
            decoder_output):
        """
        decoder_output: batch, target_len, input_dim
        """
        # get rid of eos
        output = decoder_output
        boolean_output = self.boolean_network(output)
        attr_output = self.attribute_network(output)
        #pred_mask = torch.gt(boolean_output, 0)
        #prod = attr_output[0] * pred_mask[0]
        #print(f"pred attr {prod[0:6, 0:8]}") 
        return dict(
                pred_attributes= attr_output,
                pred_mask = boolean_output
               )

    def compute_loss(self, 
                    predicted_attrs,
                    predicted_mask,
                    target_attrs,
                    mask):

        # mask out non-predicted stuff
        to_mult = mask 
        mask_binary = torch.gt(mask, 0).float()

        if self.binary:
            to_mult = mask_binary

        predicted_attrs = predicted_attrs * to_mult
        target_attrs = target_attrs * to_mult

        attr_loss = self.attr_loss_function(predicted_attrs, target_attrs) * self.loss_multiplier
        # see if annotated at all; don't model annotator confidence, already modeled above
        mask_loss = self.mask_loss_function(predicted_mask, mask_binary) * self.loss_multiplier

        predicted_attrs = predicted_attrs[mask_binary==1]
        target_attrs = target_attrs[mask_binary==1]

        flat_pred = predicted_attrs.reshape(-1).detach().cpu().numpy()
        flat_true = target_attrs.reshape(-1).detach().cpu().numpy()

        r, __ = pearsonr(flat_pred, flat_true)

        self.metrics(attr_loss.item())
        self.metrics(mask_loss.item())

        return dict(loss=attr_loss + mask_loss)
    
    @classmethod
    def from_params(cls, params, **kwargs):
        return cls(params['input_dim'], 
                   params['hidden_dim'], 
                   params['output_dim'],
                   params['n_layers'],
                   params.get("loss_multiplier", 10),
                   params.get("loss_function",  MSECrossEntropyLoss()), 
                   params.get("activation", torch.nn.ReLU()),
                   params.get("share_networks", False))
