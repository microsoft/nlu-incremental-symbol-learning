# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, List, Iterator, Any
import logging
import itertools
import glob
import os
import numpy as np

from overrides import overrides
from conllu import parse_incr


from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.universal_dependencies_multilang import get_file_paths, UniversalDependenciesMultiLangDatasetReader

from miso.data.tokenizers import AMRBertTokenizer, AMRXLMRobertaTokenizer, MisoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ud-syntax") 
class MisoUDDatasetReader(UniversalDependenciesMultiLangDatasetReader):
    def __init__(self,
                 languages: List[str],
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: MisoTokenizer = None, #AMRTransformerTokenizer,
                 use_language_specific_pos: bool = False,
                 lazy: bool = False,
                 alternate: bool = True,
                 is_first_pass_for_vocab: bool = True,
                 max_src_len: int = 75, 
                 instances_per_file: int = 32) -> None:
        super(MisoUDDatasetReader, self).__init__(languages,
                                                  source_token_indexers,
                                                  use_language_specific_pos,
                                                  lazy,
                                                  alternate,
                                                  is_first_pass_for_vocab,
                                                  instances_per_file)
        self._tokenizer = tokenizer
        self._source_token_indexers = source_token_indexers

        self._syntax_edge_type_indexers = {"syn_edge_types": SingleIdTokenIndexer(namespace="syn_edge_types")}
        self._max_src_len = max_src_len

    @overrides
    def _read_one_file(self, lang: str, file_path: str):
        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances for %s language from conllu dataset at: %s", lang, file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if x["id"] is not None]

                heads = [x["head"] for x in annotation]
                tags = [x["deprel"] for x in annotation]
                words = [x["form"] for x in annotation]
                if self._use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
                instance = self.text_to_instance(lang, words, pos_tags, list(zip(tags, heads)))
                if instance is None:
                    continue
                yield instance

    def trim_dependencies(self, deps): 
        """
        remove all relations that involve heads which are outside of max len 
        """
        new_deps = []
        for deprel, head in deps: 
            if head > self._max_src_len:
                # replace with max len 
                head = self._max_src_len
            new_deps.append((deprel, head)) 
        return new_deps[0: self._max_src_len]


    @overrides
    def text_to_instance(self,  # type: ignore
                         lang: str,
                         words: List[str],
                         upos_tags: List[str],
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        lang : ``str``, required.
            The language identifier.
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields. The language identifier is stored in the metadata.
        """
        fields: Dict[str, Field] = {}
        og_words = words 

        # trim words
        if self._max_src_len is not None and len(words) > self._max_src_len:
            #return None
            words = words[0:self._max_src_len]
            upos_tags = upos_tags[0:self._max_src_len]
            dependencies = self.trim_dependencies(dependencies) 

        # trim words
        if self._max_src_len is not None and len(words) > self._max_src_len:
            return None
            #words = words[0:self._max_src_len]
            #upos_tags = upos_tags[0:self._max_src_len]
            #dependencies = dependencies[0:self._max_src_len+1]

        if self._tokenizer is not None:
            bert_tokenizer_ret = self._tokenizer.tokenize(words, True)
            src_token_ids = bert_tokenizer_ret["token_ids"]
            src_token_subword_index = bert_tokenizer_ret["token_recovery_matrix"]
            if src_token_ids.shape[0] > 512: 
                return None
            
            
        else:
            src_token_ids, src_token_subword_index = None, None

        fields["source_tokens"] = TextField(
            tokens=[Token(x) for x in words],
            token_indexers=self._source_token_indexers
        )
        fields["source_pos_tags"] = SequenceLabelField(upos_tags, 
                                                       fields["source_tokens"], 
                                                       label_namespace="pos_tags")

        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["syn_edge_types"] = TextField(
                tokens=[Token(x[0]) for x in dependencies],
                token_indexers=self._syntax_edge_type_indexers,
            )
            try:
                fields["syn_edge_heads"] = SequenceLabelField(
                    labels=[int(x[1]) for x in dependencies],
                    sequence_field=fields["syn_edge_types"],
                    label_namespace="syn_edge_heads"
                )
            except TypeError:
                print(words)
                sys.exit() 


        if src_token_ids is not None:
            fields['source_subtoken_ids'] = ArrayField(src_token_ids)

        if src_token_subword_index is not None:
            fields['source_token_recovery_matrix'] = ArrayField(src_token_subword_index)

        fields['syn_edge_head_mask'] = ArrayField(np.ones((len(words), len(words)), dtype='uint8'))
        fields['syn_valid_node_mask'] = ArrayField(np.array([1] * len(words), dtype='uint8')) 

        fields["syn_tokens_str"] = MetadataField(
                og_words)

        fields["metadata"] = MetadataField({"syn_tokens_str": words, "src_pos_str": upos_tags, "lang": lang})

        return Instance(fields)
