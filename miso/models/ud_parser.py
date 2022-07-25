# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any
import logging
from collections import OrderedDict
import os 
import pdb 

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
from miso.modules.seq2seq_encoders.transformer_encoder import MisoTransformerEncoder
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

@Model.register("ud_parser")
class UDParser(Transduction):
    """
    Model will use only the encoder part of a Transduction model,
    but to make maximally compatible we'll have it inherit and just
    not use the decoder modules. 
    """
    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: BaseBertWrapper,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder_pos_embedding: Embedding,
                 encoder: Seq2SeqEncoder,
                 syntax_edge_type_namespace: str = None,
                 biaffine_parser: DeepTreeParser = None,
                 dropout: float = 0.0,
                 eps: float = 1e-20,
                 pretrained_weights: str = None,
                 vocab_dir: str = None,
                 ) -> None:

        super(UDParser, self).__init__(vocab=vocab,
                                       bert_encoder=bert_encoder,
                                       encoder_token_embedder=encoder_token_embedder,
                                       encoder=encoder,
                                       decoder_token_embedder=None,
                                       decoder_node_index_embedding=None,
                                       decoder=None,
                                       extended_pointer_generator=None,
                                       tree_parser=None,
                                       label_smoothing=None,
                                       target_output_namespace=None,
                                       pretrained_weights=pretrained_weights,
                                       dropout=dropout,
                                       eps=eps)
                                       

        # source-side
        self.encoder_pos_embedding=encoder_pos_embedding
        # misc
        self._syntax_edge_type_namespace=syntax_edge_type_namespace
        self.biaffine_parser = biaffine_parser
        self.vocab_dir = vocab_dir
        #metrics
        self._syntax_metrics = AttachmentScores()
        self.syntax_las = 0.0 
        self.syntax_uas = 0.0 
        # compatibility
        self.loss_mixer = None
        self.syntactic_method = "encoder-side" 
        
        # pretrained
        if self.pretrained_weights is not None:
            self.load_partial(self.pretrained_weights)
        # load vocab 
        if self.vocab_dir is not None:
            syn_vocab = Vocabulary.from_files(vocab_dir) 
            self.vocab._token_to_index[self._syntax_edge_type_namespace] = syn_vocab._token_to_index[self._syntax_edge_type_namespace]

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = OrderedDict(
            syn_uas=0.0,
            syn_las=0.0,
        )

        metrics["syn_las"] = self.syntax_las
        metrics["syn_uas"] = self.syntax_uas
        return metrics

    def _update_syntax_scores(self):
        scores = self._syntax_metrics.get_metric(reset=True)
        self.syntax_las = scores["LAS"] * 100
        self.syntax_uas = scores["UAS"] * 100

    def _compute_biaffine_loss(self, biaffine_outputs, inputs):
        #print(f"pred heads {biaffine_outputs['edge_heads']}") 
        #print(f"true heads {inputs['syn_edge_heads']}") 
        #print(f"pred tags  {biaffine_outputs['edge_types']}") 
        #print(f"true types {inputs['syn_edge_types']['syn_edge_types']}") 
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

    def _read_edge_predictions(self,
                               edge_predictions: Dict[str, torch.Tensor],
                               is_syntax = False) -> Tuple[List[List[int]], List[List[str]]]:
        edge_type_predictions = []
        edge_head_predictions = edge_predictions["edge_heads"].tolist()
        edge_type_ind_predictions = edge_predictions["edge_types"].tolist()

        if is_syntax:
            namespace = self._syntax_edge_type_namespace
        else:
            namespace = self._edge_type_namespace

        for edge_types in edge_type_ind_predictions:
            edge_type_predictions.append([
                self.vocab.get_token_from_index(edge_type, namespace) for edge_type in edge_types]
            )
        return edge_head_predictions, edge_type_predictions, edge_type_ind_predictions

    @overrides
    def _prepare_inputs(self, raw_inputs):
        inputs = raw_inputs.copy()

        inputs["source_mask"] = get_text_field_mask(raw_inputs["source_tokens"])

        source_subtoken_ids = raw_inputs.get("source_subtoken_ids", None)
        if source_subtoken_ids is None:
            inputs["source_subtoken_ids"] = None
        else:
            inputs["source_subtoken_ids"] = source_subtoken_ids.long()

        source_token_recovery_matrix = raw_inputs.get("source_token_recovery_matrix", None)
        if source_token_recovery_matrix is None:
            inputs["source_token_recovery_matrix"] = None
        else:
            inputs["source_token_recovery_matrix"] = source_token_recovery_matrix.long()

        return inputs 

    def _transformer_encode(self,
                tokens: Dict[str, torch.Tensor],
                subtoken_ids: torch.Tensor,
                token_recovery_matrix: torch.Tensor,
                mask: torch.Tensor,
                **kwargs) -> Dict:

        # [batch, num_tokens, embedding_size]
        encoder_inputs = [self._encoder_token_embedder(tokens)]
        if subtoken_ids is not None and self._bert_encoder is not None:
            bert_embeddings = self._bert_encoder(
                input_ids=subtoken_ids,
                attention_mask=subtoken_ids.ne(0),
                output_all_encoded_layers=False,
                token_recovery_matrix=token_recovery_matrix
            )
            encoder_inputs += [bert_embeddings]
        encoder_inputs = torch.cat(encoder_inputs, 2)
        encoder_inputs = self._dropout(encoder_inputs)

        # [batch, num_tokens, encoder_output_size]
        encoder_outputs = self._encoder(encoder_inputs, mask)
        encoder_outputs = self._dropout(encoder_outputs)

        return dict(
            encoder_outputs=encoder_outputs,
        )

    @overrides
    def _encode(self, inputs) -> Dict:
        if isinstance(self._encoder, MisoTransformerEncoder):
            encoding_outputs = self._transformer_encode(
                tokens=inputs["source_tokens"],
                pos_tags=inputs["source_pos_tags"],
                subtoken_ids=inputs["source_subtoken_ids"],
                token_recovery_matrix=inputs["source_token_recovery_matrix"],
                mask=inputs["source_mask"]
            )
        else:
            encoding_outputs = super()._encode(
                tokens=inputs["source_tokens"],
                pos_tags=inputs["source_pos_tags"],
                subtoken_ids=inputs["source_subtoken_ids"],
                token_recovery_matrix=inputs["source_token_recovery_matrix"],
                mask=inputs["source_mask"]
            )
        return encoding_outputs

    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(inputs) 

        biaffine_outputs = self._parse_syntax(encoding_outputs['encoder_outputs'],
                                        inputs["syn_edge_head_mask"],
                                        inputs["syn_edge_heads"],
                                        do_mst = False) 



        biaffine_loss = self._compute_biaffine_loss(biaffine_outputs,
                                                    inputs)


        return dict(loss=biaffine_loss) 

    def _test_forward(self, inputs: Dict) -> Dict:
        encoding_outputs = self._encode(inputs)

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
