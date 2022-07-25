# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Iterator, Callable, Dict
import logging
import json 
import os
import sys
from glob import glob 
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.fields import TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL

#from decomp import UDSCorpus

from miso.data.fields.continuous_label_field import ContinuousLabelField
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.data.dataset_readers.decomp_parsing.tests import DROP_TEST_CASES, NODROP_TEST_CASES, test_reader
from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
#from miso.data.dataset_readers.decomp_parsing.uds import TestUDSCorpus
from miso.data.tokenizers import AMRBertTokenizer, AMRXLMRobertaTokenizer, MisoTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("decomp")
class DecompDatasetReader(DatasetReader):
    '''
    Dataset reader for Decomp data
    '''
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer],
                 target_token_indexers: Dict[str, TokenIndexer],
                 generation_token_indexers: Dict[str, TokenIndexer],
                 tokenizer: MisoTokenizer = None, #AMRTransformerTokenizer,
                 evaluation: bool = False,
                 drop_syntax: bool = True,
                 semantics_only: bool = False,
                 line_limit: int = None,
                 order: str = "sorted",
                 lazy: bool = False,
                 api_time: bool = False,
                 ) -> None:

        super().__init__(lazy=lazy)
        self.drop_syntax = drop_syntax 
        self.semantics_only = semantics_only
        self.line_limit = line_limit
        self.order = order 

        if self.drop_syntax: 
            self.test_cases = DROP_TEST_CASES
        else:
            self.test_cases = NODROP_TEST_CASES

        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers
        self._generation_token_indexers = generation_token_indexers
        self._edge_type_indexers = {"edge_types": SingleIdTokenIndexer(namespace="edge_types")}
        self._tokenizer = tokenizer
        self._num_subtokens = 0
        self._num_subtoken_oovs = 0

        self.eval = evaluation

        self._number_bert_ids = 0
        self._number_bert_oov_ids = 0
        self._number_non_oov_pos_tags = 0
        self._number_pos_tags = 0
    
        self.over_len = 0
        self.api_time = api_time

    def report_coverage(self):
        if self._number_bert_ids != 0:
            logger.info('BERT OOV  rate: {0:.4f} ({1}/{2})'.format(
                self._number_bert_oov_ids / self._number_bert_ids,
                self._number_bert_oov_ids, self._number_bert_ids
            ))
        if self._number_non_oov_pos_tags != 0:
            logger.info('POS tag coverage: {0:.4f} ({1}/{2})'.format(
                self._number_non_oov_pos_tags / self._number_pos_tags,
                self._number_non_oov_pos_tags, self._number_pos_tags
            ))

    def set_evaluation(self):
        self.eval = True
    
    @overrides
    def _read(self, split: str) -> Iterable[Instance]:

        logger.info("Reading decompositional semantic data from: %s", split)
        if split in ['train', 'test', 'dev']:
            uds = UDSCorpus(split = split)
        else:
            # if not standard (pretraining data)
            if split.endswith(".json"):
                uds = UDSCorpus.from_json(split)
            #else:
            #    # data is just lines of input text
            #    if self.api_time:
            #        uds = TestUDSCorpus.from_single_line(split)
            #    else:
            #        uds = TestUDSCorpus.from_lines(split)

        # corpus is Graphs and annotations 
        i=0
        skipped = 0
        for name, graph in uds.graphs.items():               
            i+=1

            t2i = self.text_to_instance(graph)
            if t2i is None:
                skipped += 1
                continue
            if self.line_limit is not None:
                if i > self.line_limit:
                    break

            yield t2i

    def pprint_graph(self, graph, full_graph = 0):
        if full_graph:
            for node in graph.nodes:
                try:
                    print("{}: {}".format(node, graph.nodes[node]['form']))
                except KeyError:
                    print("{}".format(node))
            for edge in graph.edges:
                print(edge, graph.edges[edge])

        else:
            # get semantics subgraph
            sem_subgraph = graph.semantics_subgraph
            for node in sem_subgraph.nodes:
                print("node {} has span {}".format(
                    node, graph.span(node, attrs = ['form'])
                    ))
            print("edges")
            [print(e, sem_subgraph.edges[e]) for e in sem_subgraph.edges]

        
    
    def spot_check(self, graph, list_data):
        self.pprint_graph(graph, True)
        #print(" ".join(list_data['tgt_tokens']))
        #print([(x[0], x[1].keys())  for x in zip(list_data['tgt_tokens'], list_data['tgt_attributes'])])
        print(list_data['head_indices'])
        print(["{}:{}".format(list_data['tgt_tokens'][i+1], list_data['tgt_tokens'][head + 1]) for i, head in enumerate(list_data['head_indices'])])
        print(list_data['head_tags'])
        #print("===========================")
        #print()

    @overrides
    def text_to_instance(self, graph, do_print=False) -> Instance:
        """
        Does bulk of work converting a graph to an Instance of Fields 
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        max_tgt_length = None if self.eval else 60
        d = DecompGraph(graph, drop_syntax = self.drop_syntax, order = self.order)
        list_data = d.get_list_data(
             bos=START_SYMBOL, 
             eos=END_SYMBOL, 
             bert_tokenizer = self._tokenizer, 
             max_tgt_length = max_tgt_length, 
             semantics_only = self.semantics_only)
        if list_data is None:
            return None

        if do_print:
            self.spot_check(graph, list_data)


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

        # Target-side input.
        # (exclude the last one <EOS>.)
        fields["target_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"][:-1]],
            token_indexers=self._target_token_indexers
        )

        if len(list_data['tgt_tokens']) > 60:
            self.over_len += 1

        

        fields["source_pos_tags"] = SequenceLabelField(
            labels=list_data["src_pos_tags"],
            sequence_field=fields["source_tokens"],
            label_namespace="pos_tags"
        )

        if list_data["tgt_pos_tags"] is not None:
            fields["target_pos_tags"] = SequenceLabelField(
                labels=list_data["tgt_pos_tags"][:-1],
                sequence_field=fields["target_tokens"],
                label_namespace="pos_tags"
            )

        fields["target_node_indices"] = SequenceLabelField(
            labels=list_data["tgt_indices"][:-1],
            sequence_field=fields["target_tokens"],
            label_namespace="node_indices",
        )

        # Target-side output.
        # Include <BOS> here because we want it in the generation vocabulary such that
        # at the inference starting stage, <BOS> can be correctly initialized.
        fields["generation_outputs"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens_to_generate"]],
            token_indexers=self._generation_token_indexers
        )

        fields["target_copy_indices"] = SequenceLabelField(
            labels=list_data["tgt_copy_indices"],
            sequence_field=fields["generation_outputs"],
            label_namespace="target_copy_indices",
        )

        fields["target_attention_map"] = AdjacencyField(  # TODO: replace it with ArrayField.
            indices=list_data["tgt_copy_map"],
            sequence_field=fields["generation_outputs"],
            padding_value=0
        )

        # These two fields for source copy

        fields["source_copy_indices"] = SequenceLabelField(
            labels=list_data["src_copy_indices"],
            sequence_field=fields["generation_outputs"],
            label_namespace="source_copy_indices",
        )

        fields["source_attention_map"] = AdjacencyField(  # TODO: replace it with ArrayField.
            indices=list_data["src_copy_map"],
            sequence_field=TextField(
                [Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]], None
            ),
            padding_value=0
        )
        #print(list_data['src_copy_indices']) 
        #print(list_data['src_copy_map']) 

        #print(f'over textfield {[Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]]}') 

        #print(fields["source_copy_indices"]) 
        #print(fields["source_attention_map"]) 
        #sys.exit()


        # These two fields are used in biaffine parser
        fields["edge_types"] = TextField(
            tokens=[Token(x) for x in list_data["head_tags"]],
            token_indexers=self._edge_type_indexers
        )

        fields["edge_heads"] = SequenceLabelField(
            labels=list_data["head_indices"],
            sequence_field=fields["edge_types"],
            label_namespace="edge_heads"
        )

        
        if list_data.get('node_mask', None) is not None:
            # Valid nodes are 1; pads are 0.
            fields['valid_node_mask'] = ArrayField(list_data['node_mask'])

        if list_data.get('edge_mask', None) is not None:
            # A matrix of shape [num_nodes, num_nodes] where entry (i, j) is 1
            # if and only if (1) j < i and (2) j is not an antecedent of i.
            # TODO: try to remove the second constrain.
            fields['edge_head_mask'] = ArrayField(list_data['edge_mask'])

        # node attributes 
        #print(f"tgt attr {len(list_data['tgt_attributes'])}")
        #print(list_data['tgt_attributes'])
        #print(f"target tokens {len(fields['target_tokens'])}")
        #print(fields['target_tokens'])

        fields["target_attributes"] = ContinuousLabelField(
                                        labels=list_data["tgt_attributes"][:-1],
                                        sequence_field=fields["target_tokens"],
                                        ontology = NODE_ONTOLOGY)

        # edge attributes 
        fields["edge_attributes"] = ContinuousLabelField(
                        labels = list_data["edge_attributes"][:-1],
                        sequence_field = fields["target_tokens"],
                        ontology = EDGE_ONTOLOGY)

        # this field is actually needed for scoring later
        fields["graph"] = MetadataField(
            list_data['arbor_graph'])


        # Metadata fields, good for debugging
        fields["src_tokens_str"] = MetadataField(
            list_data["src_tokens"]
        )
        

        fields["tgt_tokens_str"] = MetadataField(
            list_data.get("tgt_tokens", [])
        )

        fields["src_copy_vocab"] = MetadataField(
            list_data["src_copy_vocab"]
        )

        fields["tag_lut"] = MetadataField(
            dict(pos=list_data["pos_tag_lut"])
        )

        fields["source_copy_invalid_ids"] = MetadataField(
            list_data['src_copy_invalid_ids']
        )

        fields["node_name_list"] = MetadataField(list_data['node_name_list'])
        fields["target_dynamic_vocab"] = MetadataField(dict())

        fields["instance_meta"] = MetadataField(dict(
            pos_tag_lut=list_data["pos_tag_lut"],
            source_dynamic_vocab=list_data["src_copy_vocab"],
            target_token_indexers=self._target_token_indexers,
        ))

        to_print_keys = ["target_attributes", "target_tokens"]
        to_print = {k:v for k, v in fields.items() if k in to_print_keys}

        return Instance(fields)


