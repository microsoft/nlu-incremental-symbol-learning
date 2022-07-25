# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Iterator, Callable, Dict
import logging
import json 
import os
import sys
import re 
from glob import glob 
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL

from miso.data.tokenizers import AMRBertTokenizer, AMRXLMRobertaTokenizer, MisoTokenizer
from miso.data.dataset_readers.ud_parsing.ud import UDGraph

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

#@DatasetReader.register("ud-syntax") 
class UDDatasetReader(DatasetReader):
    '''
    Dataset reader for Decomp data
    '''
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer],
                 tokenizer: MisoTokenizer = None, #AMRTransformerTokenizer,
                 evaluation: bool = False,
                 line_limit: int = None,
                 lazy: bool = False,
                 ) -> None:

        super().__init__(lazy=lazy)
        self._source_token_indexers = source_token_indexers
        self._tokenizer = tokenizer

        self.eval = evaluation
        self.line_limit = line_limit

    def set_evaluation(self):
        self.eval = True
   
    @staticmethod
    def parse_conllu_file(path): 
        with open(path) as f1:
            file_data = f1.read().strip()

        colnames = ["ID", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
        def parse_into_dict(data):
            data_to_ret = []
            for line in data:
                line = line.split("\t") 
                assert(len(line) == len(colnames))
                line_dict = {k:v for k,v in zip(colnames, line)}
                data_to_ret.append(line_dict)
            return data_to_ret
        
        raw_chunks = re.split("\n\n", file_data) 
        chunks = [] 
        for raw_chunk in raw_chunks:
            split_chunk = re.split("\n", raw_chunk)
            chunk_id = split_chunk[0]
            sent = split_chunk[1]
            data = parse_into_dict(split_chunk[2:])
            chunks.append(data)
        return chunks 

    @overrides
    def _read(self, path: str) -> Iterable[Instance]:

        logger.info("Reading UD semantic data from: %s", path)
        conllu_dicts = UDDatasetReader.parse_conllu_file(path) 
        i=0
        skipped = 0
        for conllu_dict in conllu_dicts: 
            i+=1
            t2i = self.text_to_instance(conllu_dict)
            if t2i is None:
                skipped += 1
                continue
            if self.line_limit is not None:
                if i > self.line_limit:
                    break

            yield t2i

    @overrides
    def text_to_instance(self, graph) -> Instance:
        """
        Does bulk of work converting a graph to an Instance of Fields 
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        max_tgt_length = None if self.eval else 60
        d = UDGraph(graph)
        list_data = d.get_list_data(
             bert_tokenizer = self._tokenizer)
        if list_data is None:
            return None

        # These four fields are used for seq2seq model and target side self copy
        fields["source_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers=self._source_token_indexers
        )

        if list_data['src_token_ids'] is not None:
            fields['source_subtoken_ids'] = ArrayField(list_data['src_token_ids'])
            self._number_bert_ids += len(list_data['src_token_ids'])
            self._number_bert_oov_ids += len(
                [bert_id for bert_id in list_data['src_token_ids'] if bert_id == 100])

        if list_data['src_token_subword_index'] is not None:
            fields['source_token_recovery_matrix'] = ArrayField(list_data['src_token_subword_index'])

        fields["source_pos_tags"] = SequenceLabelField(
            labels=list_data["src_pos_tags"],
            sequence_field=fields["source_tokens"],
            label_namespace="pos_tags"
        )
        fields["syn_edge_types"] = TextField(
            tokens=[Token(x) for x in list_data["syn_head_tags"]],
            token_indexers=self._syntax_edge_type_indexers,
        )

        fields["syn_edge_heads"] = SequenceLabelField(
            labels=list_data["syn_head_indices"],
            sequence_field=fields["syn_edge_types"],
            label_namespace="syn_edge_heads"
        )

        fields['syn_edge_head_mask'] = ArrayField(list_data['syn_edge_mask'])
        fields['syn_valid_node_mask'] = ArrayField(list_data['syn_node_mask'])

        fields["syn_node_name_list"] = MetadataField(
                list_data["syn_node_name_list"])

        # Metadata fields, good for debugging
        fields["src_tokens_str"] = MetadataField(
            list_data["src_tokens"]
        )
        
        return Instance(fields)

