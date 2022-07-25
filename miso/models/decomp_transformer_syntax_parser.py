# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any
import logging
from collections import OrderedDict
import pdb 

import subprocess
import math
from overrides import overrides
import torch

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
from miso.models.decomp_transformer_parser import DecompTransformerParser
from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder, BaseBertWrapper
from miso.modules.decoders import RNNDecoder, MisoTransformerDecoder, MisoDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser, DecompTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.modules.decoders.attribute_decoder import NodeAttributeDecoder 
from miso.modules.decoders.edge_decoder import EdgeAttributeDecoder 
from miso.models.decomp_syntax_parser import DecompSyntaxParser
from miso.metrics.decomp_metrics import DecompAttrMetrics
from miso.nn.beam_search import BeamSearch
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.metrics.pearson_r import pearson_r
from miso.losses.mixing import LossMixer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("decomp_transformer_syntax_parser")
class DecompTransformerSyntaxParser(DecompTransformerParser):

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
                 decoder: MisoTransformerDecoder,
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
                 loss_mixer: LossMixer = None,
                 intermediate_graph: bool = False,
                 pretrained_weights: str = None
                 ) -> None:
        super().__init__(vocab=vocab,
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
                         dropout=dropout,
                         beam_size=beam_size,
                         max_decoding_steps=max_decoding_steps,
                         eps=eps,
                         pretrained_weights=pretrained_weights)
        
        self.syntactic_method = syntactic_method
        self.biaffine_parser = biaffine_parser
        self.intermediate_graph = intermediate_graph
        self.loss_mixer = loss_mixer
        self._syntax_metrics = AttachmentScores()
        self.syntax_las = 0.0 
        self.syntax_uas = 0.0 

        if self.pretrained_weights is not None:
            self.load_partial(self.pretrained_weights)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        decomp_metrics = self._decomp_metrics.get_metric(reset) 
        syntax_metrics = self._syntax_metrics.get_metric(reset)

        metrics = OrderedDict(
            ppl=node_pred_metrics["ppl"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            tgt_copy=node_pred_metrics["tgt_copy"] * 100,
            node_pearson=decomp_metrics["node_pearson_r"],
            edge_pearson=decomp_metrics["edge_pearson_r"],
            pearson=decomp_metrics["pearson_r"],
            uas=edge_pred_metrics["UAS"] * 100,
            las=edge_pred_metrics["LAS"] * 100,
            syn_uas=syntax_metrics["UAS"] * 100,
            syn_las=syntax_metrics["LAS"] * 100,
        )
        metrics["s_f1"] = self.val_s_f1
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

        just_syntax = False
        encoder_side = False
        # if we're doing encoder-side 
        if "syn_tokens_str" in inputs.keys():
            pass

            biaffine_outputs = self._parse_syntax(encoding_outputs['encoder_outputs'],
                                            inputs["syn_edge_head_mask"],
                                            inputs["syn_edge_heads"],
                                            do_mst = False) 



            biaffine_loss = self._compute_biaffine_loss(biaffine_outputs,
                                                        inputs)

            self._update_syntax_scores()
            encoder_side=True

            if self.intermediate_graph:
                encoder_side = DecompSyntaxParser._add_biaffine_to_encoder(encoding_outputs, biaffine_outputs)

        else:
            biaffine_loss = 0.0

        if self.intermediate_graph:
            # TODO: put op vec back in 
            decoding_outputs = self._decode(
                tokens=inputs["target_tokens"],
                # op_vec=inputs["op_vec"],
                node_indices=inputs["target_node_indices"],
                pos_tags=inputs["target_pos_tags"],
                encoder_outputs=encoding_outputs["encoder_outputs"],
                source_mask=inputs["source_mask"],
                target_mask=inputs["target_mask"]
            )
        else:
            decoding_outputs = self._decode(
                tokens=inputs["target_tokens"],
                node_indices=inputs["target_node_indices"],
                pos_tags=inputs["target_pos_tags"],
                encoder_outputs=encoding_outputs["encoder_outputs"],
                source_mask=inputs["source_mask"],
                target_mask=inputs["target_mask"]
            )

        node_prediction_outputs = self._extended_pointer_generator(
            inputs=decoding_outputs["attentional_tensors"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            target_attention_weights=decoding_outputs["target_attention_weights"],
            source_attention_map=inputs["source_attention_map"],
            target_attention_map=inputs["target_attention_map"]
        )

        try:
            # compute node attributes
            node_attribute_outputs = self._node_attribute_predict(
                decoding_outputs["outputs"][:,:-1,:],
                inputs["node_attribute_truth"],
                inputs["node_attribute_mask"]
            )
        except ValueError:
            # concat-just-syntax case
            node_attribute_outputs = {"loss": 0.0,
                                      "pred_dict": {"pred_attributes": []}}

            just_syntax = True


        edge_prediction_outputs = self._parse(
            decoding_outputs["outputs"][:,:,:],
            edge_head_mask=inputs["edge_head_mask"],
            edge_heads=inputs["edge_heads"]
        )

        try:
            edge_attribute_outputs = self._edge_attribute_predict(
                    edge_prediction_outputs["edge_type_query"],
                    edge_prediction_outputs["edge_type_key"],
                    edge_prediction_outputs["edge_heads"],
                    inputs["edge_attribute_truth"],
                    inputs["edge_attribute_mask"]
                    )
        except ValueError:
            # concat-just-syntax case
            edge_attribute_outputs = {"loss": 0.0,
                                      "pred_dict": {"pred_attributes": []}}
            just_syntax = True

        node_pred_loss = self._compute_node_prediction_loss(
            prob_dist=node_prediction_outputs["hybrid_prob_dist"],
            generation_outputs=inputs["generation_outputs"],
            source_copy_indices=inputs["source_copy_indices"],
            target_copy_indices=inputs["target_copy_indices"],
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            coverage_history=decoding_outputs["coverage_history"]
            #coverage_history=None
        )
        edge_pred_loss = self._compute_edge_prediction_loss(
            edge_head_ll=edge_prediction_outputs["edge_head_ll"],
            edge_type_ll=edge_prediction_outputs["edge_type_ll"],
            pred_edge_heads=edge_prediction_outputs["edge_heads"],
            pred_edge_types=edge_prediction_outputs["edge_types"],
            gold_edge_heads=inputs["edge_heads"],
            gold_edge_types=inputs["edge_types"],
            valid_node_mask=inputs["valid_node_mask"]
        )

        if encoder_side: 
            # learn a loss ratio
            loss = self.compute_training_loss(node_pred_loss["loss_per_node"],
                                              edge_pred_loss["loss_per_node"],
                                              node_attribute_outputs["loss"],
                                              edge_attribute_outputs["loss"],
                                              biaffine_loss)
        else:
            # no biaffine loss 
            loss = node_pred_loss["loss_per_node"] + edge_pred_loss["loss_per_node"] + \
                   node_attribute_outputs['loss'] + edge_attribute_outputs['loss']

        if not just_syntax:
            # compute combined pearson 
            self._decomp_metrics(None, None, None, None, "both")

        return dict(loss=loss, 
                    node_attributes = node_attribute_outputs['pred_dict']['pred_attributes'],
                    edge_attributes = edge_attribute_outputs['pred_dict']['pred_attributes'])

    @overrides
    def _test_forward(self, inputs: Dict) -> Dict:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )
        # if we're doing encoder-side 
        if self.biaffine_parser is not None:
            biaffine_outputs = self._parse_syntax(encoding_outputs['encoder_outputs'],
                                                  inputs["syn_edge_head_mask"],
                                                  None,
                                                  valid_node_mask = inputs["syn_valid_node_mask"],
                                                  do_mst=True)
            
            if self.intermediate_graph: 
                encoding_outputs = DecompSyntaxParser._add_biaffine_to_encoder(encoding_outputs, biaffine_outputs)

        start_predictions, start_state, auxiliaries, misc = self._prepare_decoding_start_state(inputs, encoding_outputs)

        # all_predictions: [batch_size, beam_size, max_steps]
        # outputs: [batch_size, beam_size, max_steps, hidden_vector_dim]
        # log_probs: [batch_size, beam_size]

    
        all_predictions, outputs, log_probs, target_dynamic_vocabs = self._beam_search.search(
            start_predictions=start_predictions,
            start_state=start_state,
            auxiliaries=auxiliaries,
            step=lambda x, y, z: self._take_one_step_node_prediction(x, y, z, misc),
            tracked_state_name="output",
            tracked_auxiliary_name="target_dynamic_vocabs"
        )

        node_predictions, node_index_predictions, edge_head_mask, valid_node_mask = self._read_node_predictions(
            # Remove the last one because we can't get the RNN state for the last one.
            predictions=all_predictions[:, 0, :-1],
            meta_data=inputs["instance_meta"],
            target_dynamic_vocabs=target_dynamic_vocabs[0],
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"]
        )


        node_attribute_outputs = self._node_attribute_predict(
            outputs[:,:,:-1,:],
            None, None
        )

        edge_predictions = self._parse(
            rnn_outputs=outputs[:, 0],
            edge_head_mask=edge_head_mask
        )

        (edge_head_predictions, 
        edge_type_predictions, 
        edge_type_ind_predictions) = self._read_edge_predictions(edge_predictions, is_syntax=False)

        edge_attribute_outputs = self._edge_attribute_predict(
                edge_predictions["edge_type_query"],
                edge_predictions["edge_type_key"],
                edge_predictions["edge_heads"],
                None, None
                )

        edge_pred_loss = self._compute_edge_prediction_loss(
            edge_head_ll=edge_predictions["edge_head_ll"],
            edge_type_ll=edge_predictions["edge_type_ll"],
            pred_edge_heads=edge_predictions["edge_heads"],
            pred_edge_types=edge_predictions["edge_types"],
            gold_edge_heads=edge_predictions["edge_heads"],
            gold_edge_types=edge_predictions["edge_types"],
            valid_node_mask=valid_node_mask
        )

        loss = -log_probs[:, 0].sum() / edge_pred_loss["num_nodes"] + edge_pred_loss["loss_per_node"]

        if "syn_tokens_str" not in inputs:
            inputs['syn_tokens_str'] = []
            syn_edge_head_predictions, syn_edge_type_predictions, syn_edge_type_inds = [], [], []
        else:
            syn_edge_head_predictions, syn_edge_type_predictions, syn_edge_type_inds = self._read_edge_predictions(biaffine_outputs, is_syntax = True) 

        outputs = dict(
            loss=loss,
            nodes=node_predictions,
            node_indices=node_index_predictions,
            syn_nodes=inputs['syn_tokens_str'], 
            syn_edge_heads=syn_edge_head_predictions,
            syn_edge_types=syn_edge_type_predictions,
            syn_edge_type_inds=syn_edge_type_inds,
            edge_heads=edge_head_predictions,
            edge_types=edge_type_predictions,
            edge_types_inds=edge_type_ind_predictions,
            node_attributes=node_attribute_outputs['pred_dict']['pred_attributes'],
            node_attributes_mask=node_attribute_outputs['pred_dict']['pred_mask'],
            edge_attributes=edge_attribute_outputs['pred_dict']['pred_attributes'],
            edge_attributes_mask=edge_attribute_outputs['pred_dict']['pred_mask'],
        )

        return outputs

    @overrides
    def _prepare_decoding_start_state(self, inputs: Dict, encoding_outputs: Dict[str, torch.Tensor]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        batch_size = inputs["source_tokens"]["source_tokens"].size(0)
        bos = self.vocab.get_token_index(START_SYMBOL, self._target_output_namespace)
        start_predictions = inputs["source_tokens"]["source_tokens"].new_full((batch_size,), bos)

        start_state = {
            # [batch_size, *]
            "source_memory_bank": encoding_outputs["encoder_outputs"],
            "source_mask": inputs["source_mask"],
            "target_mask": inputs["target_mask"], 
            "source_attention_map": inputs["source_attention_map"],
            "target_attention_map": inputs["source_attention_map"].new_zeros(
                (batch_size, self._max_decoding_steps, self._max_decoding_steps + 1)),
            "input_history": None, 
        }

        if "op_vec" in inputs.keys() and inputs["op_vec"] is not None:
            start_state["op_vec"] = inputs["op_vec"]

        auxiliaries = {
            "target_dynamic_vocabs": inputs["target_dynamic_vocab"]
        }
        misc = {
            "batch_size": batch_size,
            "last_decoding_step": -1,  # At <BOS>, we set it to -1.
            "source_dynamic_vocab_size": inputs["source_dynamic_vocab_size"],
            "instance_meta": inputs["instance_meta"]
        }

        return start_predictions, start_state, auxiliaries, misc

    @overrides
    def _take_one_step_node_prediction(self,
                                       last_predictions: torch.Tensor,
                                       state: Dict[str, torch.Tensor],
                                       auxiliaries: Dict[str, List[Any]],
                                       misc: Dict,
                                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[Any]]]:

        inputs = self._prepare_next_inputs(
            predictions=last_predictions,
            target_attention_map=state["target_attention_map"],
            target_dynamic_vocabs=auxiliaries["target_dynamic_vocabs"],
            meta_data=misc["instance_meta"],
            batch_size=misc["batch_size"],
            last_decoding_step=misc["last_decoding_step"],
            source_dynamic_vocab_size=misc["source_dynamic_vocab_size"]
        )
    
        # TODO: HERE we go, just concatenate "inputs" to history stored in the state 
        # need a node index history and a token history 
        # no need to update history inside of _prepare_next_inputs or double-iterate 

        decoder_inputs = torch.cat([
            self._decoder_token_embedder(inputs["tokens"]),
            self._decoder_node_index_embedding(inputs["node_indices"]),
        ], dim=2)

        # if previously decoded steps, concat them in before current input 
        if state['input_history'] is not None:
            decoder_inputs = torch.cat([state['input_history'], decoder_inputs], dim = 1)

        # set previously decoded to current step  
        state['input_history'] = decoder_inputs

        if self.intermediate_graph:
            # TODO: put op vec back in 
            decoding_outputs = self._decoder.one_step_forward(
                inputs=decoder_inputs,
                source_memory_bank=state["source_memory_bank"],
                # op_vec=state["op_vec"],
                source_mask=state["source_mask"],
                decoding_step=misc["last_decoding_step"] + 1,
                total_decoding_steps=self._max_decoding_steps,
                coverage=state.get("coverage", None)
            )
        else:
            decoding_outputs = self._decoder.one_step_forward(
                inputs=decoder_inputs,
                source_memory_bank=state["source_memory_bank"],
                source_mask=state["source_mask"],
                decoding_step=misc["last_decoding_step"] + 1,
                total_decoding_steps=self._max_decoding_steps,
                coverage=state.get("coverage", None)
            )

        state['attentional_tensor'] = decoding_outputs['attentional_tensor'].squeeze(1)
        state['output'] = decoding_outputs['output'].squeeze(1)

        if decoding_outputs["coverage"] is not None:
            state["coverage"] = decoding_outputs["coverage"]


        node_prediction_outputs = self._extended_pointer_generator(
            inputs=decoding_outputs["attentional_tensor"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            target_attention_weights=decoding_outputs["target_attention_weights"],
            source_attention_map=state["source_attention_map"],
            target_attention_map=state["target_attention_map"]
        )
        log_probs = (node_prediction_outputs["hybrid_prob_dist"] + self._eps).squeeze(1).log()

        misc["last_decoding_step"] += 1

        return log_probs, state, auxiliaries

