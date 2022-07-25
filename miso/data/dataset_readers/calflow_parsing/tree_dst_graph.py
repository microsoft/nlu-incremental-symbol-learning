# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, Counter
from typing import List, Any
import pdb 
import copy 
import re 
import json
import numpy as np

import networkx as nx
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from miso.data.dataset_readers.decomp_parsing.decomp import SourceCopyVocabulary
from miso.data.dataset_readers.calflow_parsing.calflow_graph import CalFlowGraph
from dataflow.core.program import Program, Expression, BuildStructOp, ValueOp, CallLikeOp, TypeName
from dataflow.core.lispress import (parse_lispress, 
                                    program_to_lispress, 
                                    lispress_to_program,  
                                    render_compact, 
                                    render_pretty)

NOBRACK= re.compile("[\[\]]")
PROGRAM_SEP = "__StartOfProgram"
PAD_EDGE = "EDGEPAD"

class TreeDSTGraph(CalFlowGraph):
    def __init__(self, 
                src_str: str,
                tgt_str: str,
                use_program: bool = False,
                use_agent_utterance: bool = False, 
                use_context: bool = False,
                fxn_of_interest: str = None):
        super(TreeDSTGraph, self).__init__(src_str=src_str, 
                                           tgt_str=tgt_str,
                                           use_program=use_program,
                                           use_agent_utterance=use_agent_utterance,
                                           use_context=use_context,
                                           fxn_of_interest=fxn_of_interest)

