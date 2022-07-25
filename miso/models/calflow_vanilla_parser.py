# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from miso.models.calflow_parser import CalFlowParser
from typing import List, Dict, Tuple, Any 
import logging
from collections import OrderedDict

import pdb 
from overrides import overrides
import torch
import numpy as np

from allennlp.data import Token, Instance, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder,  Seq2SeqEncoder
from allennlp.nn import util
from allennlp.nn.util import  get_text_field_mask
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.models.calflow_parser import CalFlowParser
from miso.modules.seq2seq_encoders import  BaseBertWrapper
from miso.modules.decoders import  VanillaRNNDecoder
from miso.modules.generators import  PointerGenerator
from miso.modules.label_smoothing import LabelSmoothing
from miso.metrics.extended_pointer_generator_metrics import PointerGeneratorMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("vanilla_calflow_parser")
class VanillaCalFlowParser(CalFlowParser):

    def __init__(self,
                 vocab: Vocabulary,
                 # source-side
                 bert_encoder: BaseBertWrapper,
                 encoder_token_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 # target-side
                 decoder_token_embedder: TextFieldEmbedder,
                 decoder: VanillaRNNDecoder,
                 extended_pointer_generator: PointerGenerator,
                 # misc
                 label_smoothing: LabelSmoothing,
                 target_output_namespace: str,
                 dropout: float = 0.0,
                 beam_size: int = 5,
                 max_decoding_steps: int = 50,
                 eps: float = 1e-20,
                 pretrained_weights: str = None,
                 fxn_of_interest: str = None,
                 loss_weights: List = None,
                 do_train_metrics: bool = False,
                 ) -> None:
        super().__init__(vocab=vocab,
                         # source-side
                         bert_encoder=bert_encoder,
                         encoder_token_embedder=encoder_token_embedder,
                         encoder=encoder,
                         # target-side
                         decoder_token_embedder=decoder_token_embedder,
                         decoder_node_index_embedding=None,
                         decoder=decoder,
                         extended_pointer_generator=extended_pointer_generator,
                         tree_parser=None,
                         # misc
                         label_smoothing=label_smoothing,
                         target_output_namespace=target_output_namespace,
                         edge_type_namespace=None,
                         dropout=dropout,
                         beam_size=beam_size,
                         max_decoding_steps=max_decoding_steps,
                         eps=eps,
                         pretrained_weights=pretrained_weights,
                         fxn_of_interest=fxn_of_interest,
                         loss_weights=loss_weights,
                         do_train_metrics=do_train_metrics)

        self._node_pred_metrics = PointerGeneratorMetrics()
        self.oracle = False


    def forward(self, **raw_inputs: Dict) -> Dict:
        inputs = self._prepare_inputs(raw_inputs)
        if self.training or self.oracle:
            return self._training_forward(inputs)
        else:
            return self._test_forward(inputs) 

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
        if not self.oracle:
            return super().forward_on_instances(instances)

        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.decode(self(**model_input))

            instance_separated_output: List[Dict[str, np.ndarray]] = [{} for _ in dataset.instances]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        #continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    #continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        node_pred_metrics = self._node_pred_metrics.get_metric(reset)

        metrics = OrderedDict(
            ppl=node_pred_metrics["ppl"],
            node_pred=node_pred_metrics["accuracy"] * 100,
            interest_loss=node_pred_metrics["interest_loss"],
            non_interest_loss=node_pred_metrics["non_interest_loss"],
            generate=node_pred_metrics["generate"] * 100,
            src_copy=node_pred_metrics["src_copy"] * 100,
            exact_match=self.exact_match_score * 100,
            no_refer=self.no_refer_score * 100,
        )
        if self.fxn_of_interest is not None:
            additional_metrics = {f"{self.fxn_of_interest}_coarse": self.coarse_fxn_metric * 100,
                                  f"{self.fxn_of_interest}_fine": self.fine_fxn_metric * 100}
            metrics.update(additional_metrics)

        return metrics

    def _prepare_next_inputs(self,
                             predictions: torch.Tensor,
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

        def batch_index(instance_i: int) -> int:
            if predictions.size(0) == batch_size * self._beam_size:
                return instance_i // self._beam_size
            else:
                return instance_i

        token_instances = []
        for i, index in enumerate(predictions.tolist()):
            instance_meta = meta_data[batch_index(i)]
            target_dynamic_vocab = target_dynamic_vocabs[i]
            # Generation.
            if index < self._vocab_size:
                token = self.vocab.get_token_from_index(index, self._target_output_namespace)
            # Source-side copy.
            elif self._vocab_size <= index < self._vocab_size + source_dynamic_vocab_size:
                index -= self._vocab_size
                source_dynamic_vocab = instance_meta["source_dynamic_vocab"]
                token = source_dynamic_vocab.get_token_from_idx(index)
            # Target-side copy.
            else:
                index -= (self._vocab_size + source_dynamic_vocab_size)
                token = target_dynamic_vocab[index]
                node_index = index

            target_token = TextField([Token(token)], instance_meta["target_token_indexers"])
            token_instances.append(Instance({"target_tokens": target_token}))

        # Covert tokens to tensors.
        batch = Batch(token_instances)
        batch.index_instances(self.vocab)
        padding_lengths = batch.get_padding_lengths()
        tokens = {}
        for key, tensor in batch.as_tensor_dict(padding_lengths)["target_tokens"].items():
            tokens[key] = tensor.type_as(predictions)

        return dict(
            tokens=tokens,
        )

    @overrides
    def _take_one_step_node_prediction(self,
                                       last_predictions: torch.Tensor,
                                       state: Dict[str, torch.Tensor],
                                       auxiliaries: Dict[str, List[Any]],
                                       misc: Dict,
                                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, List[Any]]]:
        
        inputs = self._prepare_next_inputs(
            predictions=last_predictions,
            target_dynamic_vocabs=auxiliaries["target_dynamic_vocabs"],
            meta_data=misc["instance_meta"],
            batch_size=misc["batch_size"],
            last_decoding_step=misc["last_decoding_step"],
            source_dynamic_vocab_size=misc["source_dynamic_vocab_size"]
        )

        decoder_inputs = torch.cat([
            self._decoder_token_embedder(inputs["tokens"]),
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
            source_attention_map=state["source_attention_map"],
        )
        log_probs = (node_prediction_outputs["hybrid_prob_dist"] + self._eps).squeeze(1).log()

        misc["last_decoding_step"] += 1

        return log_probs, state, auxiliaries

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
        # Exclude unk and the last pad.
        inputs["source_attention_map"] = raw_inputs["source_attention_map"][:, 1:-1]

        inputs["source_dynamic_vocab_size"] = inputs["source_attention_map"].size(2)

        return inputs

    @overrides
    def _decode(self,
                tokens: Dict[str, torch.Tensor],
                encoder_outputs: torch.Tensor,
                hidden_states: Tuple[torch.Tensor, torch.Tensor],
                mask: torch.Tensor,
                **kwargs) -> Dict:

        # [batch, num_tokens, embedding_size]
        decoder_inputs = torch.cat([
            self._decoder_token_embedder(tokens),
        ], dim=2)
        decoder_inputs = self._dropout(decoder_inputs)

        decoder_outputs = self._decoder(
            inputs=decoder_inputs,
            source_memory_bank=encoder_outputs,
            source_mask=mask,
            hidden_state=hidden_states
        )

        return decoder_outputs

    def get_loss(self, 
                loss_per_instance: torch.Tensor, 
                contains_fxn: torch.Tensor):
        # try:
        #     non_interest_weight, interest_weight = self.loss_weights
        # except TypeError:
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

        if torch.sum(interest_loss) > 0:
            interest_loss = torch.mean(interest_loss) * interest_weight
        else:
            interest_loss = torch.sum(interest_loss)

        non_interest_loss = torch.mean(non_interest_loss) * non_interest_weight

        return interest_loss + non_interest_loss, interest_loss / interest_weight, non_interest_loss / non_interest_weight

    @overrides
    def _compute_node_prediction_loss(self,
                                      prob_dist: torch.Tensor,
                                      generation_outputs: torch.Tensor,
                                      source_copy_indices: torch.Tensor,
                                      source_dynamic_vocab_size: int,
                                      inputs: Dict,
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
        valid_source_copy_mask = (not_pad_mask & source_copy_indices.ne(1) & source_copy_indices.ne(0))  # 1 for unk; 0 for pad.
        valid_generation_mask = (~valid_source_copy_mask) & not_pad_mask
        # Prepare hybrid targets.
        _source_copy_indices = (source_copy_indices + self._vocab_size) * valid_source_copy_mask.long()

        _generation_outputs = generation_outputs * valid_generation_mask.long()
        hybrid_targets = _source_copy_indices + _generation_outputs

        # Compute loss.
        log_prob_dist = (prob_dist.view(batch_size * target_length, -1) + self._eps).log()
        flat_hybrid_targets = hybrid_targets.view(batch_size * target_length)
        loss = self._label_smoothing(log_prob_dist, flat_hybrid_targets)

        #if self.do_reweighting:
            # need to reduce loss 
        if "contains_fxn" in inputs:
            loss, interest_loss, non_interest_loss = self.get_loss(loss, 
                                                                    inputs['contains_fxn'])
        else:
            interest_loss = 0
            non_interest_loss = 0
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
            interest_loss=interest_loss,
            non_interest_loss=non_interest_loss,
            generation_outputs=_generation_outputs,
            valid_generation_mask=valid_generation_mask,
            source_copy_indices=_source_copy_indices,
            valid_source_copy_mask=valid_source_copy_mask,
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
            encoder_outputs=encoding_outputs["encoder_outputs"],
            hidden_states=encoding_outputs["final_states"],
            mask=inputs["source_mask"]
        )

        node_prediction_outputs = self._extended_pointer_generator(
            inputs=decoding_outputs["attentional_tensors"],
            source_attention_weights=decoding_outputs["source_attention_weights"],
            source_attention_map=inputs["source_attention_map"],
        )

        node_pred_loss = self._compute_node_prediction_loss(
            prob_dist=node_prediction_outputs["hybrid_prob_dist"],
            generation_outputs=inputs["generation_outputs"],
            source_copy_indices=inputs["source_copy_indices"],
            source_dynamic_vocab_size=inputs["source_dynamic_vocab_size"],
            inputs=inputs,
            source_attention_weights=decoding_outputs["source_attention_weights"],
            coverage_history=decoding_outputs["coverage_history"]
        )

        loss = node_pred_loss["loss_per_node"] 

        return dict(loss=loss,
                    prob_dist=node_prediction_outputs['hybrid_prob_dist']) 


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



        outputs = dict(
            src_str=inputs['src_tokens_str'],
            nodes=node_predictions,
        )

        return outputs
