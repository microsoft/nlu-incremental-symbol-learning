# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Dict, Optional
import copy 
import logging 

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.modules import InputVariationalDropout
from allennlp.nn.util import add_positional_features

from miso.modules.decoders.transformer.norms import Norm 
from miso.modules.decoders.transformer.attention_layers import _get_activation_fn

logger = logging.getLogger(__name__) 

class MisoTransformerEncoderLayer(torch.nn.Module, Registrable):
    """
    Modified TransformerEncoderLayer that returns attentions 
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, 
                d_model, 
                nhead, 
                norm: Norm,
                dim_feedforward=2048,
                dropout=0.1, 
                activation="relu",
                init_scale = 256):

        super(MisoTransformerEncoderLayer, self).__init__()
        self.init_scale = init_scale

        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, add_bias_kv = True)
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = copy.deepcopy(norm)
        self.norm2 = copy.deepcopy(norm)
        self.norm3 = copy.deepcopy(norm)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # initialize attention heads 
        for m in self.modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                torch.nn.init.normal_(m.bias_v, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.bias_v))
                torch.nn.init.normal_(m.bias_k, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.bias_k))
                torch.nn.init.normal_(m.in_proj_bias, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.in_proj_weight))
                torch.nn.init.normal_(m.out_proj.weight, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.out_proj.weight))

                torch.nn.init.constant_(m.in_proj_bias, 0.)
                torch.nn.init.constant_(m.out_proj.bias, 0.)

            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean = 0, std = self._get_std_from_tensor(self.init_scale, m.weight))
                torch.nn.init.constant_(m.bias, 0.)

    @staticmethod
    def _get_std_from_tensor(init_scale, tensor):
        if len(tensor.shape) > 2:
            in_d1, in_d2, out_d = tensor.shape
            in_d = in_d1 * in_d2
        else:
            in_d, out_d = tensor.shape

        # use gain to scale as in SmallInit of https://arxiv.org/pdf/1910.05895.pdf
        return (2 / (in_d + init_scale * out_d)) ** 0.5
            
    def forward(self, tgt, memory):
        pass 

@MisoTransformerEncoderLayer.register("pre_norm") 
class MisoPreNormTransformerEncoderLayer(MisoTransformerEncoderLayer):
    def __init__(self, 
                d_model, 
                n_head, 
                norm: Norm, 
                dim_feedforward=2048,
                dropout=0.1, 
                activation="relu",
                init_scale = 256):
        super(MisoPreNormTransformerEncoderLayer, self).__init__(d_model, 
                                                                 n_head, 
                                                                 norm, 
                                                                 dim_feedforward,
                                                                 dropout, 
                                                                 activation,
                                                                 init_scale)
    @overrides  
    def forward(self, src,  src_mask=None, src_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the encoder layer.
        """

        # norm before residual as in https://arxiv.org/pdf/1910.05895.pdf
        src2 = src.clone()
        src2 = self.norm1(src2)
        src2, src_attn = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

        # residual
        src = src + self.dropout1(src2)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        # residual 
        src = src + self.dropout2(src2)

        # additional norm, do only once 
        # src = self.norm3(src)

        return src, src_attn

