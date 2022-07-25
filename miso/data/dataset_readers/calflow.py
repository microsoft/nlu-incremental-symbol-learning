# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Iterator, Callable, Dict
import logging
import pdb
import numpy as np 
from overrides import overrides
import pathlib

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL


from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from miso.data.tokenizers import   MisoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("calflow")
class CalFlowDatasetReader(DatasetReader):
    '''
    Dataset reader for CalFlow data
    '''
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer],
                 target_token_indexers: Dict[str, TokenIndexer],
                 generation_token_indexers: Dict[str, TokenIndexer],
                 source_index_indexers: Dict[str, TokenIndexer] = None,
                 source_head_indexers: Dict[str, TokenIndexer] = None,
                 source_type_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: MisoTokenizer = None, #AMRTransformerTokenizer,
                 evaluation: bool = False,
                 line_limit: int = None,
                 use_program: bool = False,
                 use_agent_utterance: bool = False,
                 use_context: bool = False, 
                 fxn_of_interest: str = None,
                 lazy: bool = False,
                 do_remove_source_triggers: bool=False,   
                 ) -> None:

        super().__init__(lazy=lazy)
        self.line_limit = line_limit
        self.use_program = use_program 
        self.use_agent_utterance = use_agent_utterance
        self.fxn_of_interest = fxn_of_interest
        self.do_remove_source_triggers = do_remove_source_triggers

        try:
            assert(not(use_program and use_agent_utterance))
        except AssertionError:
            raise AssertionError("Unsupported combination of program and agent utterance! You have to pick one")
        self.use_context = use_context

        self._source_token_indexers = source_token_indexers
        self._target_token_indexers = target_token_indexers
        self._generation_token_indexers = generation_token_indexers
        self._source_index_indexers = source_index_indexers
        self._source_head_indexers = source_head_indexers
        self._source_type_indexers = source_type_indexers

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
    def _read(self, path: str) -> Iterable[Instance]:

        logger.info("Reading calflow data from: %s", path)
        skipped = 0
        source_path = path + ".src_tok"
        target_path = path + ".tgt"

        line_idx_path = path + ".idx"
        if pathlib.Path(line_idx_path).exists():
            with open(source_path) as source_f, open(target_path) as target_f, open(line_idx_path) as line_idx_f:
                for i, (src_line, tgt_line, line_idx) in enumerate(zip(source_f, target_f, line_idx_f)): 
                    graph = CalFlowGraph(src_str = src_line, 
                                        tgt_str = tgt_line,
                                        use_program = self.use_program,
                                        use_agent_utterance = self.use_agent_utterance,
                                        use_context = self.use_context,
                                        fxn_of_interest = self.fxn_of_interest,
                                        line_idx=line_idx.strip()) 

                    t2i = self.text_to_instance(graph)
                    if t2i is None:
                        skipped += 1
                        continue
                    if self.line_limit is not None:
                        if i > self.line_limit:
                            break
                    yield t2i

        else:
            with open(source_path) as source_f, open(target_path) as target_f:
                for i, (src_line, tgt_line) in enumerate(zip(source_f, target_f)): 
                    graph = CalFlowGraph(src_str = src_line, 
                                        tgt_str = tgt_line,
                                        use_program = self.use_program,
                                        use_agent_utterance = self.use_agent_utterance,
                                        use_context = self.use_context,
                                        fxn_of_interest = self.fxn_of_interest,
                                        line_idx=None) 

                    t2i = self.text_to_instance(graph)
                    if t2i is None:
                        skipped += 1
                        continue
                    if self.line_limit is not None:
                        if i > self.line_limit:
                            break
                    yield t2i

    @overrides
    def text_to_instance(self, graph, do_print=False) -> Instance:
        """
        Does bulk of work converting a graph to an Instance of Fields 
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        max_tgt_length = None if self.eval else 60
        list_data = graph.get_list_data(
             bos=START_SYMBOL, 
             eos=END_SYMBOL, 
             bert_tokenizer = self._tokenizer, 
             max_tgt_length = max_tgt_length) 
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

        if list_data['src_indices'] is not None:
            fields['source_indices'] = TextField(tokens=[Token(x) for x in list_data['src_indices']],
                                              token_indexers=self._source_index_indexers)
            fields['source_edge_heads'] = TextField(tokens=[Token(x) for x in list_data['src_edge_heads']],
                                              token_indexers=self._source_head_indexers)
            fields['source_edge_types'] = TextField(tokens=[Token(x) for x in list_data['src_edge_types']],
                                              token_indexers=self._source_type_indexers)

        # Target-side input.
        # (exclude the last one <EOS>.)
        fields["target_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"][:-1]],
            token_indexers=self._target_token_indexers
        )

        if len(list_data['tgt_tokens']) > 60:
            self.over_len += 1


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

        if self.fxn_of_interest is not None:
            fields['contains_fxn'] = ArrayField(np.array(list_data['contains_fxn']))

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


        # this field is actually needed for scoring later
        fields["calflow_graph"] = MetadataField(
            list_data['calflow_graph'])


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

        fields["line_index"] = MetadataField(
            list_data["line_idx"]
        )

        fields["node_name_list"] = MetadataField(list_data['node_name_list'])
        fields["target_dynamic_vocab"] = MetadataField(dict())

        fields["instance_meta"] = MetadataField(dict(
            source_dynamic_vocab=list_data["src_copy_vocab"],
            target_token_indexers=self._target_token_indexers,
        ))

        return Instance(fields)


