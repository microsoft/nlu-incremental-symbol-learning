# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import numpy
import subprocess
from typing import Dict, Optional, List, Tuple, Iterable
import sys
from overrides import overrides

import torch 

from allennlp.training.trainer_base import TrainerBase
from allennlp.training.metrics import AttachmentScores

from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax
from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
from miso.training.decomp_parsing_trainer import DecompTrainer 
#from miso.data.iterators.data_iterator import DecompDataIterator, DecompBasicDataIterator 
from miso.metrics.s_metric.s_metric import S, compute_s_metric
from miso.models.decomp_syntax_only_parser import DecompSyntaxOnlyParser
from miso.models.decomp_transformer_syntax_only_parser import DecompTransformerSyntaxOnlyParser
from miso.models.ud_parser import UDParser

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerBase.register("decomp_syntax_parsing")
class DecompSyntaxTrainer(DecompTrainer):
    def __init__(self,
                 validation_data_path: str,
                 validation_prediction_path: str,
                 semantics_only: bool,
                 drop_syntax: bool,
                 include_attribute_scores: bool = False,
                 warmup_epochs: int = 0,
                 syntactic_method: str = 'concat-after',
                 *args, **kwargs):
        super(DecompSyntaxTrainer, self).__init__(validation_data_path, 
                                                  validation_prediction_path,
                                                  semantics_only,
                                                  drop_syntax,
                                                  include_attribute_scores,
                                                  warmup_epochs,
                                                  *args, **kwargs)


        self.attachment_scorer = AttachmentScores()
        self.syntactic_method = syntactic_method

        if self.model.loss_mixer is not None:
            self.model.loss_mixer.update_weights(
                                            curr_epoch = 0, 
                                            total_epochs = self._num_epochs
                                                )

    def _update_attachment_scores(self, pred_instances, true_instances):
        las = []
        uas = []

        # flatten true instances 
        if self.syntactic_method.startswith("concat"):
            token_key = "tgt_tokens_str"
            head_key = "edge_heads"
            pred_label_key = "edge_types_inds"
            true_label_key = "edge_types"
            mask_key = "valid_node_mask" 
            pred_node_key = "nodes"
        else:
            token_key = "syn_tokens_str"
            head_key = "syn_edge_heads" 
            pred_label_key = "syn_edge_type_inds" 
            true_label_key = "syn_edge_types" 
            mask_key = "syn_valid_node_mask" 
            pred_node_key = "syn_nodes" 

        all_true_nodes = [true_inst for batch in true_instances for true_inst in batch[0][token_key] ]
        all_true_edge_heads = [true_inst for batch in true_instances for true_inst in batch[0][head_key] ]
        all_true_edge_types = [true_inst for batch in true_instances for true_inst in batch[0][true_label_key][true_label_key]]
        all_true_masks = [true_inst for batch in true_instances for true_inst in batch[0][mask_key]]
        assert(len(all_true_nodes) == len(all_true_edge_heads) == len(all_true_edge_types) == len(all_true_masks)  == len(pred_instances)) 

        for i in range(len(pred_instances)):
            # get rid of @start@ symbol 
            true_nodes = all_true_nodes[i]
            pred_nodes = pred_instances[i][pred_node_key]

            if self.syntactic_method.startswith("concat"): 
                if self.syntactic_method == "concat-just-syntax":
                    split_point = -1
                    end_point = min(true_nodes.index("@end@") - 1, len(pred_nodes) -1) 
                else:
                    split_point = true_nodes.index("@syntax-sep@") - 1
                    end_point = min(true_nodes.index("@end@") - 1, len(pred_nodes)-1)

            else:
                split_point = -1
                end_point = len(true_nodes) 

            try:
                pred_edge_heads = pred_instances[i][head_key][split_point + 1:end_point]
                pred_edge_types = pred_instances[i][pred_label_key][split_point+1:end_point]
            except IndexError:
                las.append(0)
                uas.append(0)
                continue

            gold_edge_heads = all_true_edge_heads[i][split_point+1:end_point]
            gold_edge_types = all_true_edge_types[i][split_point+1:end_point]
            valid_node_mask = all_true_masks[i][split_point+1:end_point]

            pred_edge_heads = torch.tensor(pred_edge_heads) 
            pred_edge_types = torch.tensor(pred_edge_types) 
            
            try:
                self.attachment_scorer(predicted_indices=pred_edge_heads,
                                                predicted_labels=pred_edge_types,
                                                gold_indices=gold_edge_heads,
                                                gold_labels=gold_edge_types,
                                                mask=valid_node_mask
                                                )
            except RuntimeError:
                continue

        scores = self.attachment_scorer.get_metric(reset = True) 
        self.model.syntax_las = scores["LAS"] * 100
        self.model.syntax_uas = scores["UAS"] * 100

    @overrides
    def _update_validation_s_score(self, pred_instances: List[Dict[str, numpy.ndarray]],
                                         true_instances):
        """Write the validation output in pkl format, and compute the S score."""
        # compute attachement scores here without having to override another function
        self._update_attachment_scores(pred_instances, true_instances) 
        
        if isinstance(self.model, DecompSyntaxOnlyParser) or \
           isinstance(self.model, DecompTransformerSyntaxOnlyParser) or \
           isinstance(self.model, UDParser):
            return 

        logger.info("Computing S")

        for batch in true_instances:
            assert(len(batch) == 1)

        true_graphs = [true_inst for batch in true_instances for true_inst in batch[0]['graph'] ]
        true_sents = [true_inst for batch in true_instances for true_inst in batch[0]['src_tokens_str']]

        pred_graphs = [DecompGraphWithSyntax.from_prediction(pred_inst, self.syntactic_method) for pred_inst in pred_instances]

        pred_sem_graphs, pred_syn_graphs, __  = zip(*pred_graphs)

        ret = compute_s_metric(true_graphs, pred_sem_graphs, true_sents, 
                               self.semantics_only, 
                               self.drop_syntax, 
                               self.include_attribute_scores)

        self.model.val_s_precision = float(ret[0]) * 100
        self.model.val_s_recall = float(ret[1]) * 100
        self.model.val_s_f1 = float(ret[2]) * 100

