# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Dict, Optional
from overrides import overrides
import numpy as np 
np.set_printoptions(precision=2, linewidth=300) 

import torch
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_log_softmax
from miso.modules.attention import Attention
from allennlp.nn.chu_liu_edmonds import decode_mst 

class DeepTreeParser(torch.nn.Module, Registrable):

    def __init__(self,
                 query_vector_dim: int,
                 key_vector_dim: int,
                 edge_head_vector_dim: int,
                 edge_type_vector_dim: int,
                 attention: Attention,
                 num_labels: int = 0,
                 dropout: float = 0.0,
                 is_syntax: bool = False) -> None:
        super().__init__()
        self.edge_head_query_linear = torch.nn.Linear(query_vector_dim, edge_head_vector_dim)
        self.edge_head_key_linear = torch.nn.Linear(key_vector_dim, edge_head_vector_dim)
        self.edge_type_query_linear = torch.nn.Linear(query_vector_dim, edge_type_vector_dim)
        self.edge_type_key_linear = torch.nn.Linear(key_vector_dim, edge_type_vector_dim)
        self.attention = attention
        self.sentinel = torch.nn.Parameter(torch.randn([1, 1, key_vector_dim]))
        self.dropout = torch.nn.Dropout2d(p=dropout)
        self.is_syntax = is_syntax

        if num_labels > 0:
            self.edge_type_bilinear = torch.nn.Bilinear(edge_type_vector_dim, edge_type_vector_dim, num_labels)
        else:
            self.edge_type_bilinear = None

        self._minus_inf = -1e8
        self._query_vector_dim = query_vector_dim
        self._key_vector_dim = key_vector_dim
        self._edge_type_vector_dim = edge_type_vector_dim

    def reset_edge_type_bilinear(self, num_labels: int) -> None:
        self.edge_type_bilinear = torch.nn.Bilinear(self._edge_type_vector_dim, self._edge_type_vector_dim, num_labels)

    def _decode_mst(self, edge_type_query, edge_type_key, edge_head_scores, mask):
        batch_size, max_length, edge_label_hidden_size = edge_type_query.size()
        lengths = mask.data.sum(dim=1).long().cpu().numpy() 
        
        #if not self.is_syntax:
        expanded_shape_query = [batch_size, max_length, max_length + 1, edge_label_hidden_size]
        expanded_shape_key = [batch_size, max_length, max_length + 1, edge_label_hidden_size]
        #else:
        #    expanded_shape_query = [batch_size, max_length, max_length, edge_label_hidden_size]
        #    expanded_shape_key = [batch_size, max_length, max_length , edge_label_hidden_size]

        edge_type_query  = edge_type_query.unsqueeze(2).expand(*expanded_shape_query).contiguous()
        edge_type_key = edge_type_key.unsqueeze(1).expand(*expanded_shape_key).contiguous()

        # [batch, max_head_length, max_modifier_length, num_labels]
        edge_type_scores = self.edge_type_bilinear(edge_type_query, edge_type_key)
        # [batch, num_labels, max_head_length, max_modifier_length]
        edge_type_scores = torch.nn.functional.log_softmax(edge_type_scores, dim=3).permute(0, 3, 1, 2)

        # Set padding positions to -inf
        minus_mask = (1 - mask.float()) * self._minus_inf

        edge_head_scores = edge_head_scores.masked_fill_(~mask.unsqueeze(2).bool(), self._minus_inf)

        # [batch, max_head_length, max_modifier_length]
        edge_head_scores = torch.nn.functional.log_softmax(edge_head_scores, dim=2)

        # [batch, num_labels, max_head_length, max_modifier_length]
        batch_energy = torch.exp(edge_head_scores.unsqueeze(1) + edge_type_scores)
        #batch_energy = torch.exp(edge_head_scores.unsqueeze(1)) 
        bsz, n_lab, seq_len, __ = batch_energy.shape

        #if not self.is_syntax: 
        sentinel = torch.zeros(bsz, n_lab, 1, seq_len + 1).to(batch_energy.device) 
        batch_energy = torch.cat([sentinel, batch_energy], dim = 2) 
        batch_energy[0,0,0,0] = 1

        batch_energy = batch_energy.permute(0,1,3,2) 
        lengths += 1
        edge_heads, edge_labels = self._run_mst_decoding(batch_energy, lengths)

        #edge_heads[edge_heads == 0] = -1
        #if not self.is_syntax: 
        edge_heads = edge_heads[:, 1:] 
        edge_labels = edge_labels[:, 1:]

        return edge_heads, edge_labels

    @staticmethod
    def _enforce_root(energy): 
        # energy: num_labels x seq_len x seq_len
        # for each i,j in seq_len
        # i to j, head to dependent 
        # so row 0 should only have 1 thing > -inf, except for 0-0 
        # num_labels x 1
        _minus_inf = -1e8
       
        ROW = True
        # get second max dependent besides 0-0 edge 
        # max over cols at row 0 
        if ROW:
            row_val, row_idx = torch.max(energy[:,0,1:], dim = 1)
            row_val = row_val.clone()
            row_idx += 1
            # wipe out row 0
            energy[:, 0, 1:] = _minus_inf
            # reset best column in row 0
            energy[:,0,row_idx] = row_val 
        else:
            col_val, col_idx = torch.max(energy[:,1:,0], dim = 1)
            col_val = col_val.clone()
            col_idx += 1
            # wipe out row 0
            energy[:, 1:, 0] = _minus_inf
            # reset best column in col 0
            energy[:,col_idx,0] = col_val 

        return energy

    @staticmethod
    def _run_mst_decoding(batch_energy, lengths):
        edge_heads = []
        edge_labels = []

        for i, (energy, length) in enumerate(zip(batch_energy.detach().cpu(), lengths)):
            # decode heads and labels 
            # need to decode labels separately so that we can enforce single root 
            scores, label_ids = energy.max(dim=0)
            energy = scores

            instance_heads, instance_head_labels = decode_mst(energy.numpy(), length, has_labels=False)
            #instance_heads, instance_head_labels = decode_mst(scores.numpy(), length, has_labels=True)

            ## Find the labels which correspond to the edges in the max spanning tree.
            instance_head_labels = []
            for child, parent in enumerate(instance_heads):
                instance_head_labels.append(label_ids[parent, child].item())
            
            # check for multiroot
            multi_root = sum([1 if h == 0 else 0 for h in instance_heads[0:length]]) > 1

            if multi_root: 
                energy = energy.unsqueeze(0)
                energy = DeepTreeParser._enforce_root(energy) 
                energy = energy.squeeze(0) 
                instance_heads, instance_head_labels = decode_mst(energy.numpy(), length, has_labels=False)
                #instance_heads, instance_head_labels = decode_mst(scores.numpy(), length, has_labels=True)

                ## Find the labels which correspond to the edges in the max spanning tree.
                instance_head_labels = []
                for child, parent in enumerate(instance_heads):
                    instance_head_labels.append(label_ids[parent, child].item())


            edge_heads.append(instance_heads)
            edge_labels.append(instance_head_labels)

        return torch.from_numpy(np.stack(edge_heads)), torch.from_numpy(np.stack(edge_labels))

    @overrides
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                edge_head_mask: torch.ByteTensor = None,
                gold_edge_heads: torch.Tensor = None,
                decode_mst: bool = False,
                valid_node_mask: torch.Tensor = None
                ) -> Dict:
        """
        :param query: [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head_mask: [batch_size, query_length, key_length]
                        1 indicates a valid position; otherwise, 0.
        :param gold_edge_heads: None or [batch_size, query_length].
                        head indices start from 1.
        :return:
            edge_heads: [batch_size, query_length].
            edge_types: [batch_size, query_length].
            edge_head_ll: [batch_size, query_length, key_length + 1(sentinel)].
            edge_type_ll: [batch_size, query_length, num_labels] (based on gold_edge_head) or None.
        """
        #if not self.is_syntax: 
        key, edge_head_mask = self._add_sentinel(query, key, edge_head_mask)

        edge_head_query, edge_head_key, edge_type_query, edge_type_key = self._mlp(query, key)

        # [batch_size, query_length, key_length + 1]
        edge_head_score = self._get_edge_head_score(edge_head_query, edge_head_key)

        if not decode_mst:
            # for pred when not using MST 
            edge_heads, edge_types = self._greedy_search(
                edge_type_query, edge_type_key, edge_head_score, edge_head_mask
            )

        else:
            #edge_heads, edge_types = self._greedy_search(
            #    edge_type_query, edge_type_key, edge_head_score, edge_head_mask
            #)

            edge_heads, edge_types = self._decode_mst(
                edge_type_query, edge_type_key, edge_head_score, valid_node_mask 
            )

        if gold_edge_heads is None:
            # test-time we don't have gold heads, use predicted heads 
            gold_edge_heads = edge_heads.long()

        # for loss 
        # [batch_size, query_length, num_labels]
        edge_type_score = self._get_edge_type_score(edge_type_query, edge_type_key, gold_edge_heads)

        edge_reps = self._get_representations(edge_head_score, edge_head_key, edge_type_key) 

        return dict(
            # Note: head indices start from 1.
            edge_heads=edge_heads,
            edge_types=edge_types,
            edge_reps=edge_reps, 
            # Log-Likelihood.
            edge_head_ll=masked_log_softmax(edge_head_score, edge_head_mask, dim=2),
            edge_type_ll=masked_log_softmax(edge_type_score, None, dim=2)
        )

    def _add_sentinel(self,
                      query: torch.FloatTensor,
                      key: torch.FloatTensor,
                      mask: torch.ByteTensor) -> Tuple:
        """
        Add a sentinel at the beginning of keys.
        :param query:  [batch_size, query_length, input_vector_dim]
        :param key:  [batch_size, key_length, key_vector_size]
        :param mask: None or [batch_size, query_length, key_length]
        :return:
            new_keys: [batch_size, key_length + 1, input_vector_dim]
            mask: None or [batch_size, query_length, key_length + 1]
        """
        batch_size, query_length, _ = query.size()
        if key is None:
            new_keys = self.sentinel.expand([batch_size, 1, self._key_vector_dim])
            new_mask = self.sentinel.new_ones(batch_size, query_length, 1)
            return new_keys, new_mask

        sentinel = self.sentinel.expand([batch_size, 1, self._key_vector_dim])
        new_keys = torch.cat([sentinel, key], dim=1)
        new_mask = None
        if mask is not None:
            sentinel_mask = mask.new_ones(batch_size, query_length, 1)
            new_mask = torch.cat([sentinel_mask, mask], dim=2)
        return new_keys, new_mask

    def _mlp(self,
             query: torch.FloatTensor,
             key: torch.FloatTensor) -> Tuple:
        """
        Transform query and key into spaces of edge and label.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :return:
            edge_head_query: [batch_size, query_length, edge_head_vector_ddim]
            edge_head_key: [batch_size, key_length, edge_head_vector_dim]
            edge_type_query: [batch_size, query_length, edge_type_vector_dim]
            edge_type_key: [batch_size, key_length, edge_type_vector_dim]
        """
        query_length = query.size(1)
        edge_head_query = F.elu(self.edge_head_query_linear(query))
        edge_head_key = F.elu(self.edge_head_key_linear(key))

        edge_type_query = F.elu(self.edge_type_query_linear(query))
        edge_type_key = F.elu(self.edge_type_key_linear(key))

        edge_head = torch.cat([edge_head_query, edge_head_key], dim=1)
        edge_type = torch.cat([edge_type_query, edge_type_key], dim=1)
        edge_head = self.dropout(edge_head.transpose(1, 2)).transpose(1, 2)
        edge_type = self.dropout(edge_type.transpose(1, 2)).transpose(1, 2)

        edge_head_query = edge_head[:, :query_length]
        edge_head_key = edge_head[:, query_length:]
        edge_type_query = edge_type[:, :query_length]
        edge_type_key = edge_type[:, query_length:]

        return edge_head_query, edge_head_key, edge_type_query, edge_type_key

    def _get_edge_head_score(self,
                             query: torch.FloatTensor,
                             key: torch.FloatTensor,
                             mask: torch.Tensor = None) -> torch.FloatTensor:
        """
        Compute the edge head scores.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key:  [batch_size, key_length, key_vector_dim]
        :param mask:  None or [batch_size, query_length, key_length]
        :return: [batch_size, query_length, key_length]
        """
        edge_head_score = self.attention(query, key).squeeze(1)
        return edge_head_score

    def _get_edge_type_score(self,
                             query: torch.FloatTensor,
                             key: torch.FloatTensor,
                             edge_head: torch.Tensor) -> torch.Tensor:
        """
        Compute the edge type scores.
        :param query:  [batch_size, query_length, query_vector_dim]
        :param key: [batch_size, key_length, key_vector_dim]
        :param edge_head: [batch_size, query_length]
        :return:
            label_score: None or [batch_size, query_length, num_labels]
        """
        batch_size = key.size(0)
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_head)
        # [batch_size, query_length, hidden_size]
        selected_key = key[batch_index, edge_head].contiguous()
        query = query.contiguous()
        edge_type_score = self.edge_type_bilinear(query, selected_key)

        return edge_type_score

    def _get_representations(self,
                             edge_head_scores: torch.FloatTensor,
                             edge_head_key: torch.FloatTensor,
                             edge_type_key: torch.FloatTensor):
        """
        Compute the weighted representations of the head and type keys 
        :param edge_head_scores:  [batch_size, query_length, key_length]
        :param edge_head_key: [batch_size, key_length, key_head_dim]
        :param edge_type_key: [batch_size, key_length, key_type_dim]
        :return:
            reps: [batch_size, query_length, key_head_dim + key_type_dim]
        """
        edge_head_scores = torch.exp(F.log_softmax(edge_head_scores, dim=2))

        # [batch_size, query_length, key_length] x [batch_size, key_length, key_head_dim] 
        # -> [batch_size, query_length, key_length]
        weighted_head_key = edge_head_scores @ edge_head_key
        # [batch_size, query_length, key_length] x [batch_size, key_length, key_type_dim] 
        # -> [batch_size, query_length, key_length]
        weighted_type_key = edge_head_scores @ edge_type_key

        keys_and_types = torch.cat([weighted_head_key, weighted_type_key], dim = 2)

        return keys_and_types

    def _greedy_search(self,
                       query: torch.FloatTensor,
                       key: torch.FloatTensor,
                       edge_head_score: torch.FloatTensor,
                       edge_head_mask: torch.ByteTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edge heads and labels.
        :param query: [batch_size, query_length, query_vector_dim]
        :param key:  [batch_size, key_length, key_vector_dim]
        :param edge_head_score:  [batch_size, query_length, key_length]
        :param edge_head_mask:  None or [batch_size, query_length, key_length]
        :return:
            edge_head: [batch_size, query_length]
            edge_type: [batch_size, query_length]
        """
        edge_head_score = edge_head_score.masked_fill_(~edge_head_mask.bool(), self._minus_inf)
        _, edge_head = edge_head_score.max(dim=2)

        edge_type_score = self._get_edge_type_score(query, key, edge_head)
        _, edge_type = edge_type_score.max(dim=2)

        return edge_head, edge_type
