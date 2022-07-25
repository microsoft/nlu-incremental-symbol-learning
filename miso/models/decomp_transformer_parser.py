# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any
import logging
from collections import OrderedDict

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

from miso.models.transduction_base import Transduction
from miso.models.decomp_parser import DecompParser 
from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder, BaseBertWrapper
from miso.modules.decoders import RNNDecoder, MisoTransformerDecoder, MisoDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser, DecompTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.modules.decoders.attribute_decoder import NodeAttributeDecoder 
from miso.modules.decoders.edge_decoder import EdgeAttributeDecoder 
from miso.metrics.decomp_metrics import DecompAttrMetrics
from miso.nn.beam_search import BeamSearch
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.metrics.pearson_r import pearson_r
# The following imports are added for mimick testing.
#from miso.predictors.predictor import Predictor
#from miso.commands.predict import _PredictManager
#from miso.commands.s_score import Scorer, compute_args, ComputeTup
#from miso.losses import LossFunctionDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("decomp_transformer_parser")
class DecompTransformerParser(DecompParser):

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
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 eps: float = 1e-20,
                 pretrained_weights: str = None,
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

    @overrides
    def _encode(self,
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
    def _decode(self,
                tokens: Dict[str, torch.Tensor],
                node_indices: torch.Tensor,
                encoder_outputs: torch.Tensor,
                source_mask: torch.Tensor,
                target_mask: torch.Tensor,
                op_vec: torch.Tensor = None,
                **kwargs) -> Dict:

        # [batch, num_tokens, embedding_size]
        decoder_inputs = torch.cat([
            self._decoder_token_embedder(tokens),
            self._decoder_node_index_embedding(node_indices),
        ], dim=2)

        decoder_inputs = self._dropout(decoder_inputs)
        if op_vec is None:
            decoder_outputs = self._decoder(
                inputs=decoder_inputs,
                source_memory_bank=encoder_outputs,
                source_mask = source_mask,
                target_mask = target_mask
            )
        else:
            decoder_outputs = self._decoder(
                inputs=decoder_inputs,
                source_memory_bank=encoder_outputs,
                op_vec=op_vec,
                source_mask = source_mask,
                target_mask = target_mask
            )


        return decoder_outputs

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


    @overrides
    def _prepare_inputs(self, raw_inputs):
        inputs = raw_inputs.copy()

        inputs["source_mask"] = get_text_field_mask(raw_inputs["source_tokens"])
        inputs["target_mask"] = get_text_field_mask(raw_inputs["target_tokens"])

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

        # Exclude <BOS>.
        inputs["generation_outputs"] = raw_inputs["generation_outputs"]["generation_tokens"][:, 1:]
        inputs["source_copy_indices"] = raw_inputs["source_copy_indices"][:, 1:]
        inputs["target_copy_indices"] = raw_inputs["target_copy_indices"][:, 1:]


        # [batch, target_seq_length, target_seq_length + 1(sentinel)]
        inputs["target_attention_map"] = raw_inputs["target_attention_map"][:, 1:]  # exclude UNK
        # [batch, 1(unk) + source_seq_length, dynamic_vocab_size]
        # Exclude unk and the last pad.
        inputs["source_attention_map"] = raw_inputs["source_attention_map"][:, 1:-1]

        inputs["source_dynamic_vocab_size"] = inputs["source_attention_map"].size(2)

        inputs["edge_types"] = raw_inputs["edge_types"]["edge_types"]

        # self._pprint(inputs)
        node_attribute_stack = raw_inputs['target_attributes']
        node_attribute_values = node_attribute_stack[0,:,:,:].squeeze(0)
        node_attribute_mask = node_attribute_stack[1,:,:,:].squeeze(0)
        edge_attribute_stack = raw_inputs['edge_attributes']
        edge_attribute_values = edge_attribute_stack[0,:,:,:].squeeze(0)
        edge_attribute_mask = edge_attribute_stack[1,:,:,:].squeeze(0)

        if len(node_attribute_values.shape) == 2:
            node_attribute_values = node_attribute_values.unsqueeze(0)
            node_attribute_mask = node_attribute_mask.unsqueeze(0)
            edge_attribute_values = edge_attribute_values.unsqueeze(0)
            edge_attribute_mask = edge_attribute_mask.unsqueeze(0)

        inputs.update(dict(
                # like decoder_token_inputs
                node_attribute_truth = node_attribute_values[:,:-1,:],
                node_attribute_mask = node_attribute_mask[:,:-1,:],
                edge_attribute_truth = edge_attribute_values[:,:-1,:],
                edge_attribute_mask = edge_attribute_mask[:,:-1,:]
        ))

        return inputs
    
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
    def _prepare_next_inputs(self,
                             predictions: torch.Tensor,
                             target_attention_map: torch.Tensor,
                             target_dynamic_vocabs: List[Dict[int, str]],
                             meta_data: List[Dict],
                             batch_size: int,
                             last_decoding_step: int,
                             source_dynamic_vocab_size: int) -> Dict:
        """
        Read out a group of hybrid predictions. Based on different ways of node prediction,
        find the corresponding token, node index and pos tags. Prepare the tensorized inputs
        for the next decoding step. Update the target attention map, target dynamic vocab, etc.
        :param predictions: [group_size,]
        :param target_attention_map: [group_size, target_length, target_dynamic_vocab_size].
        :param target_dynamic_vocabs: a group_size list of target dynamic vocabs.
        :param meta_data: meta data for each instance.
        :param batch_size: int.
        :param last_decoding_step: the decoding step starts from 0, so the last decoding step
            starts from -1.
        :param source_dynamic_vocab_size: int.
        """
        # On the default, if a new node is created via either generation or source-side copy,
        # its node index will be last_decoding_step + 1. One shift between the last decoding
        # step and the default node index is because node index 0 is reserved for no target copy.
        # See `_prepare_inputs` for detail.
        default_node_index = last_decoding_step + 1

        def batch_index(instance_i: int) -> int:
            if predictions.size(0) == batch_size * self._beam_size:
                return instance_i // self._beam_size
            else:
                return instance_i

        token_instances = []

        node_indices = torch.zeros_like(predictions)
        pos_tags = torch.zeros_like(predictions)

        for i, index in enumerate(predictions.tolist()):

            instance_meta = meta_data[batch_index(i)]
            pos_tag_lut = instance_meta["pos_tag_lut"]
            target_dynamic_vocab = target_dynamic_vocabs[i]
            # Generation.
            if index < self._vocab_size:
                token = self.vocab.get_token_from_index(index, self._target_output_namespace)
                node_index = default_node_index
                pos_tag = pos_tag_lut.get(token, DEFAULT_OOV_TOKEN)
            # Source-side copy.
            elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                index -= self._vocab_size
                source_dynamic_vocab = instance_meta["source_dynamic_vocab"]
                token = source_dynamic_vocab.get_token_from_idx(index)
                node_index = default_node_index
                pos_tag = pos_tag_lut.get(token, DEFAULT_OOV_TOKEN)
            # Target-side copy.
            else:
                index -= (self._vocab_size + source_dynamic_vocab_size)
                token = target_dynamic_vocab[index]
                node_index = index
                pos_tag = pos_tag_lut.get(token, DEFAULT_OOV_TOKEN)

            target_token = TextField([Token(token)], instance_meta["target_token_indexers"])

            token_instances.append(Instance({"target_tokens": target_token}))
            node_indices[i] = node_index
            pos_tags[i] = self.vocab.get_token_index(pos_tag, self._pos_tag_namespace)
            if last_decoding_step != -1:  # For <BOS>, we set the last decoding step to -1.
                target_attention_map[i, last_decoding_step, node_index] = 1
                target_dynamic_vocab[node_index] = token

        # Covert tokens to tensors.
        batch = Batch(token_instances)
        batch.index_instances(self.vocab)
        padding_lengths = batch.get_padding_lengths()
        tokens = {}
        for key, tensor in batch.as_tensor_dict(padding_lengths)["target_tokens"].items():
            tokens[key] = tensor.type_as(predictions)

        return dict(
            tokens=tokens,
            # [group_size, 1]
            node_indices=node_indices.unsqueeze(1),
            pos_tags=pos_tags.unsqueeze(1),
        )

    @overrides
    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            pos_tags=inputs["source_pos_tags"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )

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

        # compute node attributes
        node_attribute_outputs = self._node_attribute_predict(
            decoding_outputs["outputs"][:,:-1,:],
            inputs["node_attribute_truth"],
            inputs["node_attribute_mask"]
        )

        edge_prediction_outputs = self._parse(
            decoding_outputs["outputs"][:,:,:],
            edge_head_mask=inputs["edge_head_mask"],
            edge_heads=inputs["edge_heads"]
        )

        edge_attribute_outputs = self._edge_attribute_predict(
                edge_prediction_outputs["edge_type_query"],
                edge_prediction_outputs["edge_type_key"],
                edge_prediction_outputs["edge_heads"],
                inputs["edge_attribute_truth"],
                inputs["edge_attribute_mask"]
                )

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

        loss = node_pred_loss["loss_per_node"] + edge_pred_loss["loss_per_node"] + \
               node_attribute_outputs['loss'] + edge_attribute_outputs['loss']

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
        edge_type_ind_predictions) = self._read_edge_predictions(edge_predictions)

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

        outputs = dict(
            loss=loss,
            nodes=node_predictions,
            node_indices=node_index_predictions,
            edge_heads=edge_head_predictions,
            edge_types=edge_type_predictions,
            edge_types_inds=edge_type_ind_predictions,
            node_attributes=node_attribute_outputs['pred_dict']['pred_attributes'],
            node_attributes_mask=node_attribute_outputs['pred_dict']['pred_mask'],
            edge_attributes=edge_attribute_outputs['pred_dict']['pred_attributes'],
            edge_attributes_mask=edge_attribute_outputs['pred_dict']['pred_mask'],
        )

        return outputs
