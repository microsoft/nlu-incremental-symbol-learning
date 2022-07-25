# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Dict, Tuple, Any
import logging
import pdb 

from overrides import overrides
import torch
import numpy as np

from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, Seq2SeqEncoder
from allennlp.nn import util
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.models.calflow_parser import CalFlowParser 
from miso.modules.seq2seq_encoders import  BaseBertWrapper
from miso.modules.decoders import  MisoTransformerDecoder, MisoDecoder
from miso.modules.generators import ExtendedPointerGenerator
from miso.modules.parsers import  DecompTreeParser
from miso.modules.label_smoothing import LabelSmoothing

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("calflow_transformer_parser")
class CalFlowTransformerParser(CalFlowParser):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: BaseBertWrapper,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder_node_index_embedding: Embedding,
                 decoder: MisoTransformerDecoder,
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
                 do_group_dro: bool = False,
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
                         edge_type_namespace=edge_type_namespace,
                         encoder_index_embedder=encoder_index_embedder,
                         encoder_head_embedder=encoder_head_embedder,
                         encoder_type_embedder=encoder_type_embedder,
                         dropout=dropout,
                         beam_size=beam_size,
                         max_decoding_steps=max_decoding_steps,
                         eps=eps,
                         pretrained_weights=pretrained_weights,
                         fxn_of_interest=fxn_of_interest,
                         loss_weights=loss_weights,
                         do_train_metrics=do_train_metrics) 

        self.oracle = False
        self.top_k_beam_search = False
        self.top_k = 1
        self.do_group_dro = do_group_dro

        # make sure we never have the wrong settings
        assert(not ((self.training or self.oracle) and self.top_k_beam_search))

    @overrides
    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training or self.oracle:
            return self._training_forward(inputs)
        elif self.top_k_beam_search:
            return self._test_forward_top_k(inputs, k=self.top_k)
        else:
            return self._test_forward(inputs) 

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
            )
            if source_indices is not None: 
                # pad out bert embeddings 
                bsz, seq_len, __  = encoder_inputs[0].shape
                __, bert_seq_len, bert_size = bert_embeddings.shape
                padding = torch.zeros((bsz, seq_len - bert_seq_len, bert_size)).to(bert_embeddings.device).type(bert_embeddings.dtype)
                bert_embeddings = torch.cat([bert_embeddings, padding], dim=1)

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
        find the corresponding token and node index. Prepare the tensorized inputs
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


    def get_dro_loss(self,
                    node_loss_per_instance: torch.Tensor,
                    edge_loss_per_instance: torch.Tensor,
                    contains_fxn: torch.Tensor):
        # iterate over batch to separate out loss groups 
        # reshape loss 
        bsz, __ = contains_fxn.shape 
        total_loss = node_loss_per_instance + edge_loss_per_instance
        contains_fxn = contains_fxn.squeeze(-1)

        interest_loss = total_loss[contains_fxn == 1]
        non_interest_loss = total_loss[contains_fxn == 0]

        interest_loss_sum = torch.mean(interest_loss) 
        non_interest_loss_sum = torch.mean(non_interest_loss) 
        return torch.max(interest_loss_sum, non_interest_loss_sum)

    def _compute_edge_prediction_loss_dro(self,
                                          edge_head_ll: torch.Tensor,
                                          edge_type_ll: torch.Tensor,
                                          pred_edge_heads: torch.Tensor,
                                          pred_edge_types: torch.Tensor,
                                          gold_edge_heads: torch.Tensor,
                                          gold_edge_types: torch.Tensor,
                                          valid_node_mask: torch.Tensor,
                                          syntax: bool = False) -> Dict:
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

        pdb.set_trace() 
        gold_edge_head_ll = edge_head_ll[batch_indices, node_indices, gold_edge_heads]
        gold_edge_type_ll = edge_type_ll[batch_indices, node_indices, gold_edge_types]
        # Set the ll of invalid nodes to 0.
        num_nodes = valid_node_mask.sum().float()

        if not syntax: 
            # don't incur loss on EOS/SOS token
            valid_node_mask[gold_edge_heads == -1] = 0

        valid_node_mask = valid_node_mask.bool()
        gold_edge_head_ll.masked_fill_(~valid_node_mask, 0)
        gold_edge_type_ll.masked_fill_(~valid_node_mask, 0)

        # Negative log-likelihood.
        loss = -(gold_edge_head_ll.sum() + gold_edge_type_ll)
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

        return dict(
            loss=loss,
            num_nodes=num_nodes,
            loss_per_node=loss / num_nodes,
        )

    @overrides
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

        edge_prediction_outputs = self._parse(
            decoding_outputs["outputs"][:,:,:],
            edge_head_mask=inputs["edge_head_mask"],
            edge_heads=inputs["edge_heads"]
        )

        node_pred_loss = self._compute_node_prediction_loss(
            prob_dist=node_prediction_outputs["hybrid_prob_dist"],
            generation_outputs=inputs["generation_outputs"],
            source_copy_indices=inputs["source_copy_indices"],
            target_copy_indices=inputs["target_copy_indices"],
            inputs=inputs,
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            coverage_history=decoding_outputs["coverage_history"],
            return_instance_loss=self.do_group_dro,
            #coverage_history=None
        )

        edge_pred_loss = self._compute_edge_prediction_loss(
            edge_head_ll=edge_prediction_outputs["edge_head_ll"],
            edge_type_ll=edge_prediction_outputs["edge_type_ll"],
            pred_edge_heads=edge_prediction_outputs["edge_heads"],
            pred_edge_types=edge_prediction_outputs["edge_types"],
            gold_edge_heads=inputs["edge_heads"],
            gold_edge_types=inputs["edge_types"],
            valid_node_mask=inputs["valid_node_mask"],
            return_instance_loss=self.do_group_dro,
        )

        if not self.do_group_dro:
            loss = node_pred_loss["loss_per_node"] + edge_pred_loss["loss_per_node"] 
        else:
            loss = self.get_dro_loss(node_pred_loss["loss_per_node_per_instance"], edge_pred_loss["loss_per_node_per_instance"], inputs['contains_fxn']) 

        to_ret = dict(loss=loss)

        prob_list = []
        if self.oracle:
            prob_dist=node_prediction_outputs['hybrid_prob_dist']
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"]
            meta_data = inputs['instance_meta']
            # iterate over batch
            for i in range(prob_dist.shape[0]):
                dists = []
                for timestep in range(prob_dist.shape[1]):
                    timestep_dist = {}
                    for index in range(prob_dist.shape[-1]):
                        if index < self._vocab_size:
                            node = self.vocab.get_token_from_index(index, self._target_output_namespace)
                        elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                            node = "SourceCopy"
                        else:
                            node = "TargetCopy"
                        timestep_dist[node] = prob_dist[i, timestep, index].cpu().item()
                    dists.append(timestep_dist) 
                prob_list.append(dists)

            to_ret['prob_dist'] = prob_list
        return to_ret  

    def _test_forward_top_k(self, inputs: Dict, k: int) -> List[Dict]:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
            subtoken_ids=inputs["source_subtoken_ids"],
            token_recovery_matrix=inputs["source_token_recovery_matrix"],
            mask=inputs["source_mask"]
        )

        start_predictions, start_state, auxiliaries, misc = self._prepare_decoding_start_state(inputs, encoding_outputs)

        # all_predictions: [batch_size, beam_size, max_steps]
        # outputs: [batch_size, beam_size, max_steps, hidden_vector_dim]
        # log_probs: [batch_size, beam_size]

    
        all_predictions, beam_outputs, log_probs, target_dynamic_vocabs = self._beam_search.search(
            start_predictions=start_predictions,
            start_state=start_state,
            auxiliaries=auxiliaries,
            step=lambda x, y, z: self._take_one_step_node_prediction(x, y, z, misc),
            tracked_state_name="output",
            tracked_auxiliary_name="target_dynamic_vocabs"
        )

        all_outputs = []
        for beam_idx in range(all_predictions.shape[1]):
            node_predictions, node_index_predictions, edge_head_mask, valid_node_mask = self._read_node_predictions(
                # Remove the last one because we can't get the RNN state for the last one.
                predictions=all_predictions[:, beam_idx, :-1],
                meta_data=inputs["instance_meta"],
                target_dynamic_vocabs=target_dynamic_vocabs[beam_idx],
                source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"]
            )

            edge_predictions = self._parse(
                rnn_outputs=beam_outputs[:, beam_idx],
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

            loss = -log_probs[:, beam_idx].sum() / edge_pred_loss["num_nodes"] + edge_pred_loss["loss_per_node"]

            outputs = dict(
                loss=loss,
                src_str=inputs['src_tokens_str'],
                nodes=node_predictions,
                node_indices=node_index_predictions,
                edge_heads=edge_head_predictions,
                edge_types=edge_type_predictions,
                edge_types_inds=edge_type_ind_predictions
            )
            all_outputs.append(outputs)

        flat_outputs = dict(
                loss=0.0,
                src_str=[], 
                nodes=[], 
                node_indices=[], 
                edge_heads=[], 
                edge_types=[], 
                edge_types_inds=[], 
        )
        for output in all_outputs:
            for k in output.keys():
                flat_outputs[k] += output[k]

        return flat_outputs

    @overrides
    def _test_forward(self, inputs: Dict) -> Dict:
        encoding_outputs = self._encode(
            tokens=inputs["source_tokens"],
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


        edge_predictions = self._parse(
            rnn_outputs=outputs[:, 0],
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

    @overrides
    def forward_on_instances(self,
                             instances: List[Instance]) -> List[Dict[str, np.ndarray]]:
        """
        Takes a list of  :class:`~allennlp.data.instance.Instance`s, converts that text into
        arrays using this model's :class:`Vocabulary`, passes those arrays through
        :func:`self.forward()` and :func:`self.decode()` (which by default does nothing)
        and returns the result.  Before returning the result, we convert any
        ``torch.Tensors`` into numpy arrays and separate the
        batched output into a list of individual dicts per instance. Note that typically
        this will be faster on a GPU (and conditionally, on a CPU) than repeated calls to
        :func:`forward_on_instance`.

        Parameters
        ----------
        instances : List[Instance], required
            The instances to run the model on.

        Returns
        -------
        A list of the models output for each instance.
        """
        if not self.top_k_beam_search:
            return super().forward_on_instances(instances)

        batch_size = len(instances) * self.top_k
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))

            instance_separated_output: List[Dict[str, np.ndarray]] = [{} for _ in range(len(dataset.instances) * self.top_k)]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output
