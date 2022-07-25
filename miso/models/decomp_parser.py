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
# The following imports are added for mimick testing.
#from miso.predictors.predictor import Predictor
#from miso.commands.predict import _PredictManager
#from miso.commands.s_score import Scorer, compute_args, ComputeTup
#from miso.losses import LossFunctionDict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("decomp_parser")
class DecompParser(Transduction):

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
                         encoder=encoder,
                         # target-side
                         decoder_token_embedder=decoder_token_embedder,
                         decoder_node_index_embedding=decoder_node_index_embedding,
                         decoder=decoder,
                         extended_pointer_generator=extended_pointer_generator,
                         tree_parser=tree_parser,
                         # misc
                         label_smoothing=label_smoothing,
                         target_output_namespace=target_output_namespace,
                         dropout=dropout,
                         eps=eps,
                         pretrained_weights=pretrained_weights)

        self._decomp_metrics = DecompAttrMetrics() 
        self._node_attribute_module=node_attribute_module
        self._edge_attribute_module=edge_attribute_module
        # source-side
        self._encoder_pos_embedding = encoder_pos_embedding

        # target-side
        self._decoder_pos_embedding = decoder_pos_embedding

        # metrics
        self.val_s_f1 = .0
        self.val_s_precision = .0
        self.val_s_recall = .0

        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps

        # dynamic initialization
        self._target_output_namespace = target_output_namespace
        self._pos_tag_namespace = pos_tag_namespace
        self._edge_type_namespace = edge_type_namespace
        self._syntax_edge_type_namespace = syntax_edge_type_namespace
        self._vocab_size = self.vocab.get_vocab_size(target_output_namespace)
        self._vocab_pad_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_output_namespace)
        self._vocab_bos_index = self.vocab.get_token_index(START_SYMBOL, target_output_namespace)
        self._vocab_eos_index = self.vocab.get_token_index(END_SYMBOL, target_output_namespace)
        self._extended_pointer_generator.reset_vocab_linear(
            vocab_size=vocab.get_vocab_size(target_output_namespace),
            vocab_pad_index=self._vocab_pad_index
        )
        self._tree_parser.reset_edge_type_bilinear(num_labels=vocab.get_vocab_size(edge_type_namespace))
        self._label_smoothing.reset_parameters(pad_index=self._vocab_pad_index)
        self._beam_search = BeamSearch(self._vocab_eos_index, self._max_decoding_steps, self._beam_size)

        self.oracle = False 

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        decomp_metrics = self._decomp_metrics.get_metric(reset) 

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
        )
        metrics["s_f1"] = self.val_s_f1
        return metrics

    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training or self.oracle:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs)

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

        decoder_inputs = torch.cat([
            self._decoder_token_embedder(inputs["tokens"]),
            self._decoder_node_index_embedding(inputs["node_indices"]),
        ], dim=2)

        hidden_states = (
            # [num_layers, batch_size, hidden_vector_dim]
            state["hidden_state_1"].permute(1, 0, 2),
            state["hidden_state_2"].permute(1, 0, 2),
        )

        decoding_outputs = self._decoder.one_step_forward(
            input_tensor=decoder_inputs,
            source_memory_bank=state["source_memory_bank"],
            source_mask=state["source_mask"],
            target_memory_bank=state.get("target_memory_bank", None),
            decoding_step=misc["last_decoding_step"] + 1,
            total_decoding_steps=self._max_decoding_steps,
            input_feed=state.get("input_feed", None),
            hidden_state=hidden_states,
            coverage=state.get("coverage", None)
        )
        


        state["input_feed"] = decoding_outputs["attentional_tensor"]
        state["hidden_state_1"] = decoding_outputs["hidden_state"][0].permute(1, 0, 2)
        state["hidden_state_2"] = decoding_outputs["hidden_state"][1].permute(1, 0, 2)
        state["rnn_output"] = decoding_outputs["rnn_output"].squeeze(1)
        if decoding_outputs["coverage"] is not None:
            state["coverage"] = decoding_outputs["coverage"]
        if state.get("target_memory_bank", None) is None:
            state["target_memory_bank"] = decoding_outputs["attentional_tensor"]
        else:
            state["target_memory_bank"] = torch.cat(
                [state["target_memory_bank"], decoding_outputs["attentional_tensor"]], 1
            )

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

    def _read_node_predictions(self,
                               predictions: torch.Tensor,
                               meta_data: List[Dict],
                               target_dynamic_vocabs: List[Dict],
                               source_dynamic_vocab_size: int
                               ) -> Tuple[List[List[str]], List[List[int]], torch.Tensor, torch.Tensor]:
        """
        :param predictions: [batch_size, max_steps].
        :return:
            node_predictions: [batch_size, max_steps].
            node_index_predictions: [batch_size, max_steps].
            edge_head_mask: [batch_size, max_steps, max_steps].
            valid_node_mask: [batch_size, max_steps].
        """
        batch_size, max_steps = predictions.size()
        node_predictions = []
        node_index_predictions = []
        edge_head_mask = predictions.new_ones((batch_size, max_steps, max_steps))
        edge_head_mask = torch.tril(edge_head_mask, diagonal=-1).long()
        valid_node_mask = predictions.new_zeros((batch_size, max_steps))

        for i in range(batch_size):
            nodes = []
            node_indices = []
            instance_meta = meta_data[i]
            source_dynamic_vocab = instance_meta["source_dynamic_vocab"]
            target_dynamic_vocab = target_dynamic_vocabs[i]
            prediction_list = predictions[i].tolist()
            for j, index in enumerate(prediction_list):
                if index == self._vocab_eos_index:
                    break
                valid_node_mask[i, j] = 1
                if index < self._vocab_size:
                    node = self.vocab.get_token_from_index(index, self._target_output_namespace)
                    node_index = j
                elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                    node = source_dynamic_vocab.get_token_from_idx(index - self._vocab_size)
                    node_index = j
                else:
                    node = target_dynamic_vocab[index - self._vocab_size - source_dynamic_vocab_size]
                    target_dynamic_vocab_index = index - self._vocab_size - source_dynamic_vocab_size
                    # Valid target_dynamic_vocab_index starts from 1; 0 is reserved for sentinel.
                    # Minus 1 to ensure node indices created here are consistent with node indices
                    # created by pre-defined vocab and source-side copy.
                    node_index = target_dynamic_vocab_index - 1
                    for k, prev_node_index in enumerate(node_indices):
                        if node_index == prev_node_index:
                            edge_head_mask[i, j, k] = 0
                nodes.append(node)
                node_indices.append(node_index)
            node_predictions.append(nodes)
            node_index_predictions.append(node_indices)

        return node_predictions, node_index_predictions, edge_head_mask, valid_node_mask
    

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

    def _prepare_decoding_start_state(self, inputs: Dict, encoding_outputs: Dict[str, torch.Tensor]) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        batch_size = inputs["source_tokens"]["source_tokens"].size(0)
        bos = self.vocab.get_token_index(START_SYMBOL, self._target_output_namespace)
        start_predictions = inputs["source_tokens"]["source_tokens"].new_full((batch_size,), bos)
        start_state = {
            # [batch_size, *]
            "source_memory_bank": encoding_outputs["encoder_outputs"],
            "hidden_state_1": encoding_outputs["final_states"][0].permute(1, 0, 2),
            "hidden_state_2": encoding_outputs["final_states"][1].permute(1, 0, 2),
            "source_mask": inputs["source_mask"],
            "source_attention_map": inputs["source_attention_map"],
            "target_attention_map": inputs["source_attention_map"].new_zeros(
                (batch_size, self._max_decoding_steps, self._max_decoding_steps + 1))
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

        #print(f"prepared first input hidden_state_1 {start_state['hidden_state_1'].shape}")
        #print(f"prepared first input hidden_state_2 {start_state['hidden_state_2'].shape}")
        return start_predictions, start_state, auxiliaries, misc

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

    def _node_attribute_predict(self, rnn_outputs, tgt_attr, tgt_attr_mask):
        pred_dict = self._node_attribute_module(rnn_outputs)
        if tgt_attr is not None:
            pred_attrs = pred_dict["pred_attributes"].clone()
            pred_mask = pred_dict["pred_mask"].clone()
            tgt_attr_copy = tgt_attr.detach().clone()
            tgt_mask_copy = tgt_attr_mask.detach().clone()

            mask_binary = torch.gt(tgt_mask_copy, 0)
            target_attrs = tgt_attr[mask_binary==1]
 
            flat_true = target_attrs.reshape(-1).detach().cpu().numpy()

            loss = self._node_attribute_module.compute_loss(pred_dict["pred_attributes"],
                                                            pred_dict["pred_mask"],
                                                            tgt_attr, 
                                                            tgt_attr_mask)

            self._decomp_metrics(pred_attrs,
                                 pred_mask,
                                 tgt_attr_copy, 
                                 tgt_mask_copy,
                                 "node"
                                 )



            loss = loss['loss']
        else:
            loss = -1.0000
        return {"pred_dict": pred_dict, "loss": loss}

    def _edge_attribute_predict(self, query, 
                                      key, 
                                      edge_heads, 
                                      tgt_attr,
                                      tgt_attr_mask):

        batch_size = key.size(0)
        batch_index = torch.arange(0, batch_size).view(batch_size, 1).type_as(edge_heads)
        # [batch_size, query_length, hidden_size]
        selected_key = key[batch_index, edge_heads].contiguous()
        query = query.contiguous()

        pred_dict = self._edge_attribute_module(query, selected_key)

        if tgt_attr is not None:
            self._decomp_metrics(pred_dict["pred_attributes"],
                                 pred_dict["pred_mask"],
                                 tgt_attr, 
                                 tgt_attr_mask,
                                 "edge"
                                 )

            loss = self._edge_attribute_module.compute_loss(pred_dict["pred_attributes"],
                                                            pred_dict["pred_mask"],
                                                            tgt_attr,
                                                            tgt_attr_mask)


            loss = loss['loss']
        else:
            loss = -1.0000

        return {"pred_dict": pred_dict, "loss": loss}

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
            hidden_states=encoding_outputs["final_states"],
            mask=inputs["source_mask"]
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
            decoding_outputs["rnn_outputs"][:,:-1,:],
            inputs["node_attribute_truth"],
            inputs["node_attribute_mask"]
        )

        edge_prediction_outputs = self._parse(
            rnn_outputs=decoding_outputs["rnn_outputs"],
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
        # rnn_outputs: [batch_size, beam_size, max_steps, hidden_vector_dim]
        # log_probs: [batch_size, beam_size]

    
        all_predictions, rnn_outputs, log_probs, target_dynamic_vocabs = self._beam_search.search(
            start_predictions=start_predictions,
            start_state=start_state,
            auxiliaries=auxiliaries,
            step=lambda x, y, z: self._take_one_step_node_prediction(x, y, z, misc),
            tracked_state_name="rnn_output",
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
            rnn_outputs[:,:,:-1,:],
            None, None
        )

        edge_predictions = self._parse(
            # Remove the first RNN state because it represents <BOS>.
            rnn_outputs=rnn_outputs[:, 0],
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

    def compute_training_loss(self, node_loss, edge_loss, node_attr_loss, edge_attr_loss, biaffine_loss):
        sem_loss = node_loss + edge_loss + node_attr_loss + edge_attr_loss
        syn_loss = biaffine_loss

        if self.loss_mixer is not None:
            return self.loss_mixer(sem_loss, syn_loss) 

        # default to 1-to-1 weighting 
        return sem_loss + syn_loss
