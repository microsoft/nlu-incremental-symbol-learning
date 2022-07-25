# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterable, Iterator, Callable, Dict
import logging
import pdb
from overrides import overrides
import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import  Token
from allennlp.data.fields import Field, TextField, ArrayField, SequenceLabelField, MetadataField, AdjacencyField
from allennlp.data.instance import Instance
from allennlp.common.util import START_SYMBOL, END_SYMBOL


from miso.data.dataset_readers.calflow_parsing.calflow_presence import CalFlowPresence
from miso.data.tokenizers import MisoTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("calflow_presence")
class PresenceCalFlowReader(DatasetReader):
    '''
    Dataset reader for CalFlow data to see if it contains each 
    '''
    def __init__(self,
                 source_token_indexers: Dict[str, TokenIndexer],
                 fxn_file: str,
                 use_agent_utterance: bool = False,
                 use_context: bool = False, 
                 fxn_of_interest: str = None, 
                 lazy: bool = False,
                 line_limit: int = None,
                 tokenizer: MisoTokenizer = None, #AMRTransformerTokenizer,
                 ) -> None:

        super().__init__(lazy=lazy)

        self._source_token_indexers = source_token_indexers

        self.use_agent_utterance = use_agent_utterance
        self.use_context = use_context
        self.fxn_of_interest = fxn_of_interest
        self.all_fxns = [x.strip() for x in open(fxn_file).read().split(",")]
        self.line_limit = line_limit 
        self.bert_tokenizer = tokenizer

    def set_evaluation(self):
        self.eval = True
    
    @overrides
    def _read(self, path: str) -> Iterable[Instance]:

        logger.info("Reading calflow data from: %s", path)
        skipped = 0
        source_path = path + ".src_tok"
        target_path = path + ".tgt"

        with open(source_path) as source_f, open(target_path) as target_f:
            for i, (src_line, tgt_line) in enumerate(zip(source_f, target_f)): 
                graph = CalFlowPresence(src_str = src_line, 
                                        tgt_str = tgt_line,
                                        all_fxns=self.all_fxns,
                                        use_agent_utterance = self.use_agent_utterance,
                                        use_context = self.use_context,
                                        fxn_of_interest= self.fxn_of_interest) 
                t2i = self.text_to_instance(graph)
                if t2i is None:
                    skipped += 1
                    continue
                if self.line_limit is not None:
                    if i > self.line_limit:
                        break
                yield t2i


    @overrides
    def text_to_instance(self, sequence: CalFlowPresence) -> Instance:
        """
        Does bulk of work converting a line to an Instance of Fields for the vanilla seq2seq model 
        """
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        list_data = sequence.get_list_data(self.bert_tokenizer)

        # These four fields are used for seq2seq model and target side self copy
        fields["source_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers=self._source_token_indexers
        )

        fields["contains_each"] = ArrayField(np.array(list_data['contains_each']))

        if self.fxn_of_interest is not None:
            fields['contains_fxn'] = ArrayField(np.array(list_data['contains_fxn']))
 

        # Metadata fields, good for debugging
        fields["src_tokens_str"] = MetadataField(
            list_data["src_tokens"]
        )

        fields["tgt_tokens_inputs"] = MetadataField(
            list_data.get("tgt_str", [])
        )

        fields['src_token_ids'] = ArrayField(list_data['src_token_ids'])

        return Instance(fields)


