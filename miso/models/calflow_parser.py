# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any
import logging
from collections import OrderedDict

import pdb 
from overrides import overrides
import torch

from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, Seq2SeqEncoder
from allennlp.nn.util import  get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.models.transduction_base import Transduction
from miso.modules.seq2seq_encoders import  BaseBertWrapper
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import  DecompTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.nn.beam_search import BeamSearch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("calflow_parser")
class CalFlowParser(Transduction):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: BaseBertWrapper,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder_node_index_embedding: Embedding,
                 decoder: RNNDecoder,
                 extended_pointer_generator: ExtendedPointerGenerator,
                 tree_parser: DecompTreeParser,
                 # misc
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 edge_type_namespace: str,
                 encoder_index_embedder: TextFieldEmbedder = None,
                 encoder_head_embedder: TextFieldEmbedder = None,
                 encoder_type_embedder: TextFieldEmbedder = None,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 eps: float = 1e-20,
                 pretrained_weights: str = None,
                 fxn_of_interest: str = None,
                 loss_weights: List[float] = None,
                 do_train_metrics: bool = False,
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

        self.fxn_of_interest = fxn_of_interest
        # metrics
        self.exact_match_score = .0
        self.no_refer_score = .0
        if self.fxn_of_interest is not None:
            self.coarse_fxn_metric = .0
            self.fine_fxn_metric = .0

        self._beam_size = beam_size
        self._max_decoding_steps = max_decoding_steps

        self._encoder_index_embedder = encoder_index_embedder
        self._encoder_head_embedder = encoder_head_embedder
        self._encoder_type_embedder = encoder_type_embedder
        # dynamic initialization
        self._target_output_namespace = target_output_namespace
        self._edge_type_namespace = edge_type_namespace
        self._vocab_size = self.vocab.get_vocab_size(target_output_namespace)
        self._vocab_pad_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_output_namespace)
        self._vocab_bos_index = self.vocab.get_token_index(START_SYMBOL, target_output_namespace)
        self._vocab_eos_index = self.vocab.get_token_index(END_SYMBOL, target_output_namespace)
        self._extended_pointer_generator.reset_vocab_linear(
            vocab_size=vocab.get_vocab_size(target_output_namespace),
            vocab_pad_index=self._vocab_pad_index
        )
        try:
            self._tree_parser.reset_edge_type_bilinear(num_labels=vocab.get_vocab_size(edge_type_namespace))
        except AttributeError:
            # we are secretly a Vanilla parser 
            pass 
        self._label_smoothing.reset_parameters(pad_index=self._vocab_pad_index)
        self._beam_search = BeamSearch(self._vocab_eos_index, self._max_decoding_steps, self._beam_size)

        self.oracle = False 

        self.do_reweighting = loss_weights is not None
        self.do_train_metrics = do_train_metrics
        self.loss_weights = loss_weights 

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)

        metrics = OrderedDict(
            ppl=node_pred_metrics["ppl"],
            interest_loss=node_pred_metrics["interest_loss"],
            non_interest_loss=node_pred_metrics["non_interest_loss"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            tgt_copy=node_pred_metrics["tgt_copy"] * 100,
            exact_match=self.exact_match_score * 100,
            no_refer=self.no_refer_score * 100,
            uas=edge_pred_metrics["UAS"] * 100,
            las=edge_pred_metrics["LAS"] * 100,
        )
        if self.fxn_of_interest is not None:
            additional_metrics = {f"{self.fxn_of_interest}_coarse": self.coarse_fxn_metric * 100,
                                  f"{self.fxn_of_interest}_fine": self.fine_fxn_metric * 100}
            metrics.update(additional_metrics)

        return metrics

    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training:
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

        source_indices = raw_inputs.get("source_indices", None)
        if source_indices is not None:
            inputs["source_indices"] = raw_inputs["source_indices"]
            inputs["source_edge_heads"] = raw_inputs["source_edge_heads"]
            inputs["source_edge_types"] = raw_inputs["source_edge_types"]
        else:
            inputs["source_indices"] = None
            inputs["source_edge_heads"] = None
            inputs["source_edge_types"] = None

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
        for i, index in enumerate(predictions.tolist()):
            instance_meta = meta_data[batch_index(i)]
            target_dynamic_vocab = target_dynamic_vocabs[i]
            # Generation.
            if index < self._vocab_size:
                token = self.vocab.get_token_from_index(index, self._target_output_namespace)
                node_index = default_node_index
            # Source-side copy.
            elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                index -= self._vocab_size
                source_dynamic_vocab = instance_meta["source_dynamic_vocab"]
                token = source_dynamic_vocab.get_token_from_idx(index)
                node_index = default_node_index
            # Target-side copy.
            else:
                index -= (self._vocab_size + source_dynamic_vocab_size)
                token = target_dynamic_vocab[index]
                node_index = index

            target_token = TextField([Token(token)], instance_meta["target_token_indexers"])
            token_instances.append(Instance({"target_tokens": target_token}))
            node_indices[i] = node_index
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
        )

    @overrides
    def _encode(self,
                tokens: Dict[str, torch.Tensor],
                subtoken_ids: torch.Tensor,
                token_recovery_matrix: torch.Tensor,
                mask: torch.Tensor,
                source_indices: Dict[str, torch.Tensor] = None,
                source_edge_heads: Dict[str, torch.Tensor] = None,
                source_edge_types: Dict[str, torch.Tensor] = None,
                **kwargs) -> Dict:
        # [batch, num_tokens, embedding_size]
        encoder_inputs = [self._encoder_token_embedder(tokens)]

        if source_indices is not None: 
            index_inputs = [self._encoder_index_embedder(source_indices)]
            head_inputs = [self._encoder_head_embedder(source_edge_heads)]
            type_inputs = [self._encoder_type_embedder(source_edge_types)]


        if subtoken_ids is not None and self._bert_encoder is not None:
            bert_embeddings = self._bert_encoder(
                input_ids=subtoken_ids,
                attention_mask=subtoken_ids.ne(0),
                output_all_encoded_layers=False,
                token_recovery_matrix=token_recovery_matrix
            ).detach()

            if source_indices is not None: 
                # pad out bert embeddings 
                bsz, seq_len, __  = encoder_inputs[0].shape
                __, bert_seq_len, bert_size = bert_embeddings.shape
                padding = torch.zeros((bsz, seq_len - bert_seq_len, bert_size)).to(bert_embeddings.device).type(bert_embeddings.dtype)
                bert_embeddings = torch.cat([bert_embeddings, padding], dim=1)



            encoder_inputs += [bert_embeddings]

        if source_indices is not None: 
            encoder_inputs += index_inputs + head_inputs + type_inputs 

        encoder_inputs = torch.cat(encoder_inputs, 2)
        encoder_inputs = self._dropout(encoder_inputs)

        # [batch, num_tokens, encoder_output_size]
        encoder_outputs = self._encoder(encoder_inputs, mask)
        encoder_outputs = self._dropout(encoder_outputs)
        # A tuple of (state, memory) with shape [num_layers, batch, encoder_output_size]
        encoder_final_states = self._encoder.get_final_states()
        self._encoder.reset_states()

        return dict(
            encoder_outputs=encoder_outputs,
            final_states=encoder_final_states
        )

    @overrides
    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"],
            source_indices=inputs["source_indices"],
            source_edge_heads=inputs["source_edge_heads"],
            source_edge_types=inputs["source_edge_types"]
        )


        decoding_outputs = self._decode(
            tokens=inputs["target_tokens"],
            node_indices=inputs["target_node_indices"],
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

        edge_prediction_outputs = self._parse(
            rnn_outputs=decoding_outputs["rnn_outputs"],
            edge_head_mask=inputs["edge_head_mask"],
            edge_heads=inputs["edge_heads"]
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

        loss = node_pred_loss["loss_per_node"] + edge_pred_loss["loss_per_node"] 


        return dict(loss=loss) 

    @overrides
    def _test_forward(self, inputs: Dict) -> Dict:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"],
            source_indices=inputs["source_indices"],
            source_edge_heads=inputs["source_edge_heads"],
            source_edge_types=inputs["source_edge_types"]
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

        edge_predictions = self._parse(
            # Remove the first RNN state because it represents <BOS>.
            rnn_outputs=rnn_outputs[:, 0],
            edge_head_mask=edge_head_mask
        )

        (edge_head_predictions, 
        edge_type_predictions, 
        edge_type_ind_predictions) = self._read_edge_predictions(edge_predictions)

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
            src_str=inputs['src_tokens_str'],
            nodes=node_predictions,
            node_indices=node_index_predictions,
            edge_heads=edge_head_predictions,
            edge_types=edge_type_predictions,
            edge_types_inds=edge_type_ind_predictions
        )

        return outputs

    def get_loss(self, 
                loss_per_instance: torch.Tensor, 
                contains_fxn: torch.Tensor):
        try:
            non_interest_weight, interest_weight = self.loss_weights
        except TypeError:
            non_interest_weight, interest_weight = 1, 1
        # iterate over batch to separate out loss groups 
        # reshape loss 
        bsz, __ = contains_fxn.shape 
        len_times_batch = loss_per_instance.shape[0]
        assert(len_times_batch % bsz) == 0
        seq_len = int(len_times_batch/bsz)
        loss_per_instance = loss_per_instance.reshape(bsz, seq_len, -1)
        vocab_size = loss_per_instance.shape[-1]
        contains_fxn = contains_fxn.unsqueeze(-1).repeat(1, seq_len, vocab_size)

        interest_loss = loss_per_instance[contains_fxn == 1]
        non_interest_loss = loss_per_instance[contains_fxn == 0]

        interest_loss = torch.sum(interest_loss) * interest_weight
        non_interest_loss = torch.sum(non_interest_loss) * non_interest_weight

        return interest_loss + non_interest_loss, interest_loss / interest_weight, non_interest_loss / non_interest_weight, loss_per_instance

    @overrides
    def _compute_node_prediction_loss(self,
                                      prob_dist: torch.Tensor,
                                      generation_outputs: torch.Tensor,
                                      source_copy_indices: torch.Tensor,
                                      target_copy_indices: torch.Tensor,
                                      inputs: Dict,
                                      source_dynamic_vocab_size: int,
                                      source_attention_weights: torch.Tensor = None,
                                      coverage_history: torch.Tensor = None,
                                      return_instance_loss: bool = False) -> Dict:
        """
        Compute the node prediction loss based on the final hybrid probability distribution.

        :param prob_dist: probability distribution,
            [batch_size, target_length, vocab_size + source_dynamic_vocab_size + target_dynamic_vocab_size].
        :param generation_outputs: generated node indices in the pre-defined vocabulary,
            [batch_size, target_length].
        :param source_copy_indices:  source-side copied node indices in the source dynamic vocabulary,
            [batch_size, target_length].
        :param target_copy_indices:  target-side copied node indices in the source dynamic vocabulary,
            [batch_size, target_length].
        :param source_dynamic_vocab_size: int.
        :param source_attention_weights: None or [batch_size, target_length, source_length].
        :param coverage_history: None or a tensor recording the source-side coverage history.
            [batch_size, target_length, source_length].
        """
        _, prediction = prob_dist.max(2)

        batch_size, target_length = prediction.size()
        not_pad_mask = generation_outputs.ne(self._vocab_pad_index)
        num_nodes = not_pad_mask.sum()
        num_nodes_per_instance = not_pad_mask.sum(dim=1)

        # Priority: target_copy > source_copy > generation
        # Prepare mask.
        valid_target_copy_mask = target_copy_indices.ne(0) & not_pad_mask  # 0 for sentinel.
        valid_source_copy_mask = (~valid_target_copy_mask & not_pad_mask &
                                  source_copy_indices.ne(1) & source_copy_indices.ne(0))  # 1 for unk; 0 for pad.
        valid_generation_mask = ~(valid_target_copy_mask | valid_source_copy_mask) & not_pad_mask
        # Prepare hybrid targets.
        _target_copy_indices = ((target_copy_indices + self._vocab_size + source_dynamic_vocab_size) *
                                valid_target_copy_mask.long())
        _source_copy_indices = (source_copy_indices + self._vocab_size) * valid_source_copy_mask.long()

        _generation_outputs = generation_outputs * valid_generation_mask.long()
        hybrid_targets = _target_copy_indices + _source_copy_indices + _generation_outputs

        # Compute loss.
        log_prob_dist = (prob_dist.view(batch_size * target_length, -1) + self._eps).log()
        flat_hybrid_targets = hybrid_targets.view(batch_size * target_length)

        loss = self._label_smoothing(log_prob_dist, flat_hybrid_targets)

        if self.do_reweighting or self.do_train_metrics or return_instance_loss: 
            # need to reduce loss 
            (loss, 
            loss_of_interest, 
            loss_of_noninterest,
            loss_per_instance) = self.get_loss(loss, 
                                                 inputs['contains_fxn'])

            if return_instance_loss:
                loss_per_instance = loss_per_instance.sum(dim=(1,2))
                loss_per_node_per_instance = loss_per_instance / num_nodes_per_instance



            # Coverage loss.
            if coverage_history is not None:
                #coverage_loss = torch.sum(torch.min(coverage_history.unsqueeze(-1), source_attention_weights), 2)
                coverage_loss = torch.sum(torch.min(coverage_history, source_attention_weights), 2)
                coverage_loss = (coverage_loss * not_pad_mask.float()).sum()
                loss = loss + coverage_loss

            # Update metric stats.
            self._node_pred_metrics(loss=loss,
                                    interest_loss=loss_of_interest/num_nodes,
                                    non_interest_loss=loss_of_noninterest/num_nodes,
                                    prediction=prediction,
                                    generation_outputs=_generation_outputs,
                                    valid_generation_mask=valid_generation_mask,
                                    source_copy_indices=_source_copy_indices,
                                    valid_source_copy_mask=valid_source_copy_mask,
                                    target_copy_indices=_target_copy_indices,
                                    valid_target_copy_mask=valid_target_copy_mask)

            return dict(
                loss=loss,
                loss_per_node_per_instance=loss_per_node_per_instance,
                num_nodes=num_nodes,
                loss_per_node=loss / num_nodes,
            )

        return dict(
            loss=loss,
            num_nodes=num_nodes,
            loss_per_node=loss / num_nodes,
        )