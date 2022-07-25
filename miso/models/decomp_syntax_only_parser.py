# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any
import logging
from collections import OrderedDict

import subprocess
import math
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.training.metrics import AttachmentScores

from miso.models.transduction_base import Transduction
from miso.models.decomp_syntax_parser import DecompSyntaxParser
from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder, BaseBertWrapper
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser, DecompTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.modules.decoders.attribute_decoder import NodeAttributeDecoder 
from miso.modules.decoders.edge_decoder import EdgeAttributeDecoder 
from miso.metrics.decomp_metrics import DecompAttrMetrics
from miso.nn.beam_search import BeamSearch
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.metrics.pearson_r import pearson_r
from miso.models.decomp_parser import DecompParser 
from miso.losses.mixing import LossMixer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("decomp_syntax_only_parser")
class DecompSyntaxOnlyParser(DecompSyntaxParser):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: BaseBertWrapper,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder_pos_embedding: Embedding,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder_node_index_embedding: Embedding,
                 decoder_pos_embedding: Embedding,
                 decoder: RNNDecoder,
                 extended_pointer_generator: ExtendedPointerGenerator,
                 tree_parser: DecompTreeParser,
                 node_attribute_module: NodeAttributeDecoder,
                 edge_attribute_module: EdgeAttributeDecoder,
                 # misc
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 pos_tag_namespace: str,
                 edge_type_namespace: str,
                 syntax_edge_type_namespace: str = None,
                 biaffine_parser: DeepTreeParser = None,
                 syntactic_method: str = None,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 eps: float = 1e-20,
                 loss_mixer: LossMixer = None
                 ) -> None:

        super(DecompSyntaxOnlyParser, self).__init__(
                                 vocab=vocab,
                                 # source-side
                                 bert_encoder=bert_encoder,
                                 encoder_token_embedder=encoder_token_embedder,
                                 encoder_pos_embedding=encoder_pos_embedding,
                                 encoder=encoder,
                                 # target-side
                                 decoder_token_embedder=decoder_token_embedder,
                                 decoder_node_index_embedding=decoder_node_index_embedding,
                                 decoder_pos_embedding=decoder_pos_embedding,
                                 decoder=decoder,
                                 extended_pointer_generator=extended_pointer_generator,
                                 tree_parser=tree_parser,
                                 node_attribute_module=node_attribute_module,
                                 edge_attribute_module=edge_attribute_module,
                                 # misc
                                 label_smoothing=label_smoothing,
                                 target_output_namespace=target_output_namespace,
                                 pos_tag_namespace=pos_tag_namespace,
                                 edge_type_namespace=edge_type_namespace,
                                 syntax_edge_type_namespace=syntax_edge_type_namespace,
                                 syntactic_method=syntactic_method,
                                 dropout=dropout,
                                 beam_size=beam_size,
                                 max_decoding_steps=max_decoding_steps,
                                 eps=eps)

        self.syntactic_method = "encoder-side" 
        self.biaffine_parser = biaffine_parser
        self.loss_mixer = None
        self._syntax_metrics = AttachmentScores()
        self.syntax_las = 0.0 
        self.syntax_uas = 0.0 

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        syntax_metrics = self._syntax_metrics.get_metric(reset)

        metrics = OrderedDict(
            syn_uas=syntax_metrics["UAS"] * 100,
            syn_las=syntax_metrics["LAS"] * 100,
        )
        metrics["syn_las"] = self.syntax_las
        metrics["syn_uas"] = self.syntax_uas
        return metrics

    def _update_syntax_scores(self):
        scores = self._syntax_metrics.get_metric(reset=True)
        self.syntax_las = scores["LAS"] * 100
        self.syntax_uas = scores["UAS"] * 100

    def _compute_biaffine_loss(self, biaffine_outputs, inputs):
        edge_prediction_loss = self._compute_edge_prediction_loss(
                                biaffine_outputs['edge_head_ll'],
                                biaffine_outputs['edge_type_ll'],
                                biaffine_outputs['edge_heads'],
                                biaffine_outputs['edge_types'],
                                inputs['syn_edge_heads'],
                                inputs['syn_edge_types']['syn_edge_types'],
                                inputs['syn_valid_node_mask'],
                                syntax=True)
        return edge_prediction_loss['loss_per_node']

    def _parse_syntax(self,
                      encoder_outputs: torch.Tensor,
                      edge_head_mask: torch.Tensor,
                      edge_heads: torch.Tensor = None, 
                      valid_node_mask: torch.Tensor = None,
                      do_mst = False) -> Dict:

        parser_outputs = self.biaffine_parser(
                                query=encoder_outputs,
                                key=encoder_outputs,
                                edge_head_mask=edge_head_mask,
                                gold_edge_heads=edge_heads,
                                decode_mst = do_mst,
                                valid_node_mask = valid_node_mask
                            )

        return parser_outputs

    @overrides
    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )

        biaffine_outputs = self._parse_syntax(encoding_outputs['encoder_outputs'],
                                        inputs["syn_edge_head_mask"],
                                        inputs["syn_edge_heads"],
                                        do_mst = False) 



        biaffine_loss = self._compute_biaffine_loss(biaffine_outputs,
                                                    inputs)

        self._update_syntax_scores()

        return dict(loss=biaffine_loss) 

    @overrides
    def _test_forward(self, inputs: Dict) -> Dict:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )

        biaffine_outputs = self._parse_syntax(encoding_outputs['encoder_outputs'],
                                                inputs["syn_edge_head_mask"],
                                                None,
                                                valid_node_mask = inputs["syn_valid_node_mask"],
                                                do_mst=True)

    
        syn_edge_head_predictions, syn_edge_type_predictions, syn_edge_type_inds = self._read_edge_predictions(biaffine_outputs, is_syntax = True) 

        bsz, __ = inputs["source_tokens"]["source_tokens"].shape

        outputs = dict(
            syn_nodes=inputs['syn_tokens_str'], 
            syn_edge_heads=syn_edge_head_predictions, 
            syn_edge_types=syn_edge_type_predictions,
            syn_edge_type_inds=syn_edge_type_inds,
            loss=torch.tensor([0.0]),
            nodes=torch.ones((bsz,1)),
            node_indices=torch.ones((bsz,1)),
            edge_heads=torch.ones((bsz,1)),
            edge_types=torch.ones((bsz,1)),
            edge_types_inds=torch.ones((bsz,1)),
            node_attributes=torch.ones((bsz,1,44)),
            node_attributes_mask=torch.ones((bsz,1,44)),
            edge_attributes=torch.ones((bsz,1,14)),
            edge_attributes_mask=torch.ones((bsz,1,14))
        )


        return outputs
