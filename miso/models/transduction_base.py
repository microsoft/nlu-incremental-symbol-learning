# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple
import logging
from collections import OrderedDict
import pdb 

from overrides import overrides
import torch
from torch.nn import functional as F

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, InputVariationalDropout, Seq2SeqEncoder
from allennlp.training.metrics import AttachmentScores
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

from miso.modules.seq2seq_encoders import Seq2SeqBertEncoder
from miso.modules.decoders import RNNDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import DeepTreeParser
from miso.modules.label_smoothing import LabelSmoothing
from miso.metrics.extended_pointer_generator_metrics import ExtendedPointerGeneratorMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# debugging
from icecream import ic

class Transduction(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: Seq2SeqBertEncoder,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder_node_index_embedding: Embedding,
                 decoder: RNNDecoder,
                 extended_pointer_generator: ExtendedPointerGenerator,
                 tree_parser: DeepTreeParser,
                 # misc
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 dropout: float = 0.0,
                 eps: float = 1e-20,
                 pretrained_weights: str = None,
                 ) -> None:
        super().__init__(vocab=vocab)
        # source-side
        self._bert_encoder = bert_encoder
        self._encoder_token_embedder = encoder_token_embedder
        self._encoder = encoder

        # target-side
        self._decoder_token_embedder = decoder_token_embedder
        self._decoder_node_index_embedding = decoder_node_index_embedding
        self._decoder = decoder
        self._extended_pointer_generator = extended_pointer_generator
        self._tree_parser = tree_parser

        # metrics
        self._node_pred_metrics = ExtendedPointerGeneratorMetrics()
        self._edge_pred_metrics = AttachmentScores()
        self._synt_edge_pred_metrics = AttachmentScores()

        self._label_smoothing = label_smoothing
        self._dropout = InputVariationalDropout(p=dropout)
        self._eps = eps

        # dynamic initialization
        self._target_output_namespace = target_output_namespace
        self._vocab_size = self.vocab.get_vocab_size(target_output_namespace)
        self._vocab_pad_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, target_output_namespace)

        # loading partial weights
        self.pretrained_weights = pretrained_weights

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)
        edge_pred_metrics = self._edge_pred_metrics.get_metric(reset)
        synt_edge_pred_metrics = self._synt_edge_pred_metrics.get_metric(reset)
        metrics = OrderedDict(
            ppl=node_pred_metrics["ppl"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            tgt_copy=node_pred_metrics["tgt_copy"] * 100,
            uas=edge_pred_metrics["UAS"] * 100,
            las=edge_pred_metrics["LAS"] * 100,
        )
        return metrics

    @overrides
    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs)

    def _compute_edge_prediction_loss(self,
                                      edge_head_ll: torch.Tensor,
                                      edge_type_ll: torch.Tensor,
                                      pred_edge_heads: torch.Tensor,
                                      pred_edge_types: torch.Tensor,
                                      gold_edge_heads: torch.Tensor,
                                      gold_edge_types: torch.Tensor,
                                      valid_node_mask: torch.Tensor,
                                      syntax: bool = False,
                                      return_instance_loss: bool = False) -> Dict:
        """
        Compute the edge prediction loss.

        :param edge_head_ll: [batch_size, target_length, target_length + 1 (for sentinel)].
        :param edge_type_ll: [batch_size, target_length, num_labels].
        :param pred_edge_heads: [batch_size, target_length].
        :param pred_edge_types: [batch_size, target_length].
        :param gold_edge_heads: [batch_size, target_length].
        :param gold_edge_types: [batch_size, target_length].
        :param valid_node_mask: [batch_size, target_length].
        """

        # Index the log-likelihood (ll) of gold edge heads and types.
        batch_size, target_length, _ = edge_head_ll.size()
        batch_indices = torch.arange(0, batch_size).view(batch_size, 1).type_as(gold_edge_heads)
        node_indices = torch.arange(0, target_length).view(1, target_length) \
            .expand(batch_size, target_length).type_as(gold_edge_heads)

        gold_edge_head_ll = edge_head_ll[batch_indices, node_indices, gold_edge_heads]
        gold_edge_type_ll = edge_type_ll[batch_indices, node_indices, gold_edge_types]
        # Set the ll of invalid nodes to 0.
        num_nodes = valid_node_mask.sum().float()
        num_nodes_per_instance = valid_node_mask.sum(dim=1).float() 

        if not syntax: 
            # don't incur loss on EOS/SOS token
            valid_node_mask[gold_edge_heads == -1] = 0

        valid_node_mask = valid_node_mask.bool()
        gold_edge_head_ll.masked_fill_(~valid_node_mask, 0)
        gold_edge_type_ll.masked_fill_(~valid_node_mask, 0)

        # Negative log-likelihood.

        if return_instance_loss:
            loss_per_instance = -(gold_edge_head_ll + gold_edge_type_ll)
            loss = loss_per_instance.sum()
            loss_sanity_check = -(gold_edge_head_ll.sum() + gold_edge_type_ll.sum())
            try:
                assert(abs(loss.item() - loss_sanity_check.item()) < 0.1)
            except AssertionError:
                pdb.set_trace() 

        else:
            loss = -(gold_edge_head_ll.sum() + gold_edge_type_ll.sum())

        # Update metrics.
        if self.training and not syntax:
            self._edge_pred_metrics(
                predicted_indices=pred_edge_heads,
                predicted_labels=pred_edge_types,
                gold_indices=gold_edge_heads,
                gold_labels=gold_edge_types,
                mask=valid_node_mask
            )

        elif self.training and syntax:
            self._syntax_metrics(
                predicted_indices=pred_edge_heads,
                predicted_labels=pred_edge_types,
                gold_indices=gold_edge_heads,
                gold_labels=gold_edge_types,
                mask=valid_node_mask
            )
        if return_instance_loss:
            loss_to_ret = loss 
            loss_per_node_per_instance = loss_per_instance.sum(dim=1) / num_nodes_per_instance
            return dict(
                loss=loss_to_ret,
                num_nodes=num_nodes,
                loss_per_node=loss / num_nodes,
                loss_per_node_per_instance = loss_per_node_per_instance
            )
        else:
            loss_to_ret = loss 
            return dict(
                loss=loss_to_ret,
                num_nodes=num_nodes,
                loss_per_node=loss / num_nodes,
            )

    def _compute_node_prediction_loss(self,
                                      prob_dist: torch.Tensor,
                                      generation_outputs: torch.Tensor,
                                      source_copy_indices: torch.Tensor,
                                      target_copy_indices: torch.Tensor,
                                      source_dynamic_vocab_size: int,
                                      source_attention_weights: torch.Tensor = None,
                                      coverage_history: torch.Tensor = None) -> Dict:
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

        # Coverage loss.
        if coverage_history is not None:
            #coverage_loss = torch.sum(torch.min(coverage_history.unsqueeze(-1), source_attention_weights), 2)
            coverage_loss = torch.sum(torch.min(coverage_history, source_attention_weights), 2)
            coverage_loss = (coverage_loss * not_pad_mask.float()).sum()
            loss = loss + coverage_loss

        # Update metric stats.
        self._node_pred_metrics(
            loss=loss,
            prediction=prediction,
            generation_outputs=_generation_outputs,
            valid_generation_mask=valid_generation_mask,
            source_copy_indices=_source_copy_indices,
            valid_source_copy_mask=valid_source_copy_mask,
            target_copy_indices=_target_copy_indices,
            valid_target_copy_mask=valid_target_copy_mask
        )

        return dict(
            loss=loss,
            num_nodes=num_nodes,
            loss_per_node=loss / num_nodes,
        )

    def _decode(self,
                tokens: Dict[str, torch.Tensor],
                node_indices: torch.Tensor,
                encoder_outputs: torch.Tensor,
                hidden_states: Tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor,
                **kwargs) -> Dict:

        # [batch, num_tokens, embedding_size]
        decoder_inputs = torch.cat([
            self._decoder_token_embedder(tokens),
            self._decoder_node_index_embedding(node_indices),
        ], dim=2)
        decoder_inputs = self._dropout(decoder_inputs)

        decoder_outputs = self._decoder(
            inputs=decoder_inputs,
            source_memory_bank=encoder_outputs,
            source_mask=mask,
            hidden_state=hidden_states
        )

        return decoder_outputs

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
            ).detach()

            encoder_inputs += [bert_embeddings]

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

    def _parse(self,
               rnn_outputs: torch.Tensor,
               edge_head_mask: torch.Tensor,
               edge_heads: torch.Tensor = None) -> Dict:
        """
        Based on the vector representation for each node, predict its head and edge type.
        :param rnn_outputs: vector representations of nodes, including <BOS>.
            [batch_size, target_length + 1, hidden_vector_dim].
        :param edge_head_mask: mask used in the edge head search.
            [batch_size, target_length, target_length].
        :param edge_heads: None or gold head indices, [batch_size, target_length]
        """
        # Exclude <BOS>.
        # <EOS> is already excluded in ``_prepare_inputs''.
        rnn_outputs = self._dropout(rnn_outputs[:, 1:])
        parser_outputs = self._tree_parser(
            query=rnn_outputs,
            key=rnn_outputs,
            edge_head_mask=edge_head_mask,
            gold_edge_heads=edge_heads
        )
        return parser_outputs

    def _prepare_inputs(self, raw_inputs: Dict) -> Dict:
        return raw_inputs

    def _test_forward(self, inputs: Dict) -> Dict:
        raise NotImplementedError

    def _training_forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
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

    def load_partial(self, param_file: str): 
        """
        loads weights and matches the ones it can 
        """
        logger.info(f"Attempting to load pretrained weights from {param_file}") 
        pretrained_state_dict = torch.load(param_file)
        current_state_dict = self.state_dict() 
        for k, v in pretrained_state_dict.items():
            if isinstance(v, torch.nn.Parameter):
                v = v.data
            try:
                current_state_dict[k].copy_(v)
                logger.info(f"matched {k}")
            except RuntimeError:
                new_shape = pretrained_state_dict[k].shape
                og_shape = current_state_dict[k].shape
                logger.warning(f"Unable to match {k} due to shape error: pretrained: {new_shape} vs original: {og_shape}") 
                continue 
            except KeyError:
                logger.warning(f"Unable to match {k} because it does not exist in original model") 
                continue

        key = "biaffine_parser.edge_type_query_linear.weight"
        self.load_state_dict(current_state_dict) 

