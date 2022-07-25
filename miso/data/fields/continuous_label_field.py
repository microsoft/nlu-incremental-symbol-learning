# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Union, Set, Tuple, TypeVar
import logging
import textwrap

from overrides import overrides
import torch
import numpy as np
import json

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.fields.field import Field, DataArray
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ContinuousLabelField(Field[torch.Tensor]):
    """
    A ``ContinuousLabelField`` assigns a vector of continuous labels to each element in a
    :class:`~miso.data.fields.sequence_field.SequenceField`. Used for UDSv1.0 attribute parsing,
    it automatically takes into account annotator confidence and contains both attribute values and 
    attribute masks (whether a particular attribute applies to a node).
    Because it's a labeling of some other field, we take that field as input here, and we use it to
    determine our padding and other things.


    Parameters
    ----------
    labels: ArrayField
        A sequence of continuous labels, 
    sequence_field : ``SequenceField``
        A field containing the sequence that this ``ContinuousLabelField`` is labeling.  Most often, this is a
        ``TextField``, for tagging individual tokens in a sentence.
    """
    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.

    def __init__(self,
                 labels: List[Dict],
                 sequence_field: SequenceField,
                 ontology: set) -> None:
        # sort ontology
        self.ontology = sorted(list(set(ontology)))
        self.ontology_to_idx = {}
        self.idx_to_ontology = {}
        for i, k in enumerate(self.ontology):
            self.ontology_to_idx[k] = i
            self.idx_to_ontology[i] = k

        self.labels = [None for i in range(sequence_field.sequence_length())]
        self.masks = [None for i in range(sequence_field.sequence_length())]
        for lab_idx, label_dict in enumerate(labels):
            mask_vector = np.zeros((len(self.ontology)))
            label_vector = np.zeros((len(self.ontology)))
            for k,v in label_dict.items():
                # skip everything that isn't in the ontology
                if k not in self.ontology:
                    continue
                k_idx = self.ontology_to_idx[k]
                value = v['value']
                confidence = v['confidence']
                label_vector[k_idx] = value
                mask_vector[k_idx] = confidence

            self.labels[lab_idx] = label_vector
            self.masks[lab_idx] = mask_vector

        self.sequence_field = sequence_field

        if len(labels) != sequence_field.sequence_length():
            raise ConfigurationError("Label length and sequence length "
                                     "don't match: %d and %d" % (len(labels), sequence_field.sequence_length()))


    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self.sequence_field.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        empty_vector = np.zeros((len(self.ontology)))
        desired_num_tokens = padding_lengths['num_tokens']
        padded_tags = pad_sequence_to_length(self.labels, desired_num_tokens, default_value = lambda: empty_vector)
        padded_masks = pad_sequence_to_length(self.masks, desired_num_tokens, default_value = lambda: empty_vector)
        tensor = torch.FloatTensor(padded_tags)
        mask = torch.FloatTensor(padded_masks)
        assert(tensor.shape == mask.shape)
        return tensor, mask

    @overrides
    def batch_tensors(self, tensor_list: List[Tuple[DataArray]]) -> DataArray:
        attribute_tensors, mask_tensors = zip(*tensor_list)
        stack = torch.stack((torch.stack(attribute_tensors), torch.stack(mask_tensors)))
        # concat attributes and mask 
        return torch.stack((torch.stack(attribute_tensors), torch.stack(mask_tensors)))

    @overrides
    def empty_field(self) -> 'ContinuousLabelField':  # pylint: disable=no-self-use
        # pylint: disable=protected-access
        # The empty_list here is needed for mypy
        empty_list: List[str] = []
        sequence_label_field = ContinuousLabelField(empty_list, self.sequence_field.empty_field(), set())
        sequence_label_field._indexed_labels = empty_list
        return sequence_label_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        
        content = [False for x in range(length)]
        for i, lab in enumerate(self.labels):
            if sum(lab) != 0:
                content[i] = True
        
        seq = []
        for lab in self.labels:
            as_dict = {k:v for k,v in zip(self.ontology, lab)} 
            seq.append(json.dumps(as_dict))
        content = "\n\t\t".join(seq) 
        return f"ContinuousLabelField of length {length} with content \n\t\t {content} "
