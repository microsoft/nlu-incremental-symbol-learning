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
from dataflow.core.program import Program, Expression, BuildStructOp, ValueOp, CallLikeOp, TypeName
from dataflow.core.lispress import (parse_lispress, 
                                    program_to_lispress, 
                                    lispress_to_program,  
                                    render_compact, 
                                    render_pretty)

NOBRACK= re.compile("[\[\]]")
PROGRAM_SEP = "__StartOfProgram"
PAD_EDGE = "EDGEPAD"

class CalFlowGraph:
    def __init__(self, 
                src_str: str,
                tgt_str: str,
                use_program: bool = False,
                use_agent_utterance: bool = False, 
                use_context: bool = False,
                fxn_of_interest: str = None,
                line_idx: int = None):
        #self.program = self.tgt_str_to_program(tgt_str)
        self.src_str = src_str.strip() 
        self.tgt_str = tgt_str
        self.line_idx = line_idx
        self.use_program = use_program
        self.use_agent_utterance = use_agent_utterance
        self.use_context = use_context
        self.fxn_of_interest = fxn_of_interest

        self.node_name_list = []
        self.node_idx_list  = []
        self.edge_type_list = []
        self.edge_head_list  = []
        self.parent = -1 
        self.argn = 0
        self.n_reentrant = 0

        self.node_idx_to_expr_idx = {}
        self.expr_idx_to_node_idx = {}
        self.lispress = parse_lispress(tgt_str)
        self.program, __ = lispress_to_program(self.lispress, 0)
        self.dep_chart = self.build_program_dependency_chart(self.program)
        self.fill_lists_from_program(self.program) 

        #self.prediction_to_program(self.node_name_list, self.node_idx_list, self.edge_head_list, self.edge_type_list)

        #self.ast = self.get_ast(self.tgt_str.split(" "))
        #self.preorder_ast_traversal(self.ast, parent = -1, is_function = True, argn=-1)   

    def build_program_dependency_chart(self, program):
        dep_chart = nx.DiGraph()
        for e in program.expressions:
            eid = int(NOBRACK.sub("",e.id))
            dep_chart.add_node(eid)
            for i, child_id in enumerate(e.arg_ids):
                child_id = int(NOBRACK.sub("",child_id))
                dep_chart.add_edge(eid,child_id, order_idx=i)
        return dep_chart

    def fill_lists_from_program(self, program: Program):            
        def get_arg_num(eid):
            parent_edges = list(self.dep_chart.in_edges(eid))
            parent_edge = parent_edges[-1]

            edge_order_idx = self.dep_chart.edges[parent_edge]['order_idx']
            return edge_order_idx



        # node lookup dict takes expression IDs and turns them into node ids of their parent nodes 
        parent_node_lookup = defaultdict(list)
        program_new = copy.deepcopy(program)

        # helper function for adding type_args recursively
        def add_type_name(type_name, parent_nidx, arg_idx):
            curr_idx = self.node_idx_list[-1] + 1
            self.node_idx_list.append(curr_idx) 
            self.node_name_list.append(type_name.base)
            self.edge_head_list.append(parent_nidx)
            self.edge_type_list.append(f"type_arg-{arg_idx}")
            for i, ta in enumerate(type_name.type_args):
                add_type_name(ta, curr_idx, i)

        # rules for adding 
        def add_struct_op(e: Expression, eidx: int, nidx: int):
            id = int(NOBRACK.sub("",e.id))
            try:
                parent_node_idx, argn = parent_node_lookup[id][-1]
            except (KeyError, IndexError) as ex:
                # add edge to dummy root 
                parent_node_idx, argn = 0, 0
                parent_node_lookup[id].append((parent_node_idx + self.n_reentrant, argn))

            # add node idx, with repeats for re-entrancy
            for i, (parent_node_idx, reent_argn) in enumerate(parent_node_lookup[id]):
                self.node_idx_list.append(nidx)
                # add schema 
                self.node_name_list.append(e.op.op_schema) 
                # add an edge from the expression to its parent 
                self.edge_head_list.append(parent_node_idx)

                try:
                    fxn_argn = get_arg_num(id)
                except IndexError:
                    fxn_argn = 0

                self.edge_type_list.append(f"fxn_arg-{fxn_argn}")
                # update lookup 
                self.node_idx_to_expr_idx[nidx] = eidx
                self.expr_idx_to_node_idx[eidx] = nidx
                # increment counter so that edge indices are correct 
                if i > 0:
                    self.n_reentrant += 1

            # add type args
            type_args = e.type_args
            if type_args is not None:
                for i, ta in enumerate(type_args): 
                    add_type_name(ta, nidx + self.n_reentrant, i)

            op_fields = e.op.op_fields
            ## add op nodes only once 
            try:
                assert(len(op_fields) == len(e.arg_ids))
            except AssertionError:
                assert(not e.op.empty_base)
                # if there's a base, prepend "base" to op fields 
                op_fields = ["nonEmptyBase"] + op_fields
                assert(len(op_fields) == len(e.arg_ids))

            fields_added = 0
            for i, (field, dep) in enumerate(zip(op_fields, e.arg_ids)):
                if field is not None:
                    field_node_idx = nidx + fields_added + 1
                    fields_added += 1
                    self.node_idx_list.append(field_node_idx)
                    self.node_name_list.append(field)
                    self.edge_head_list.append(nidx + self.n_reentrant)
                    self.edge_type_list.append(f"arg-{i}")

                arg_id = int(NOBRACK.sub("", dep))
                parent_node_lookup[arg_id].append((nidx+ self.n_reentrant, 0))

        def add_value_op(e: Expression, eidx: int, nidx: int):
            id = int(NOBRACK.sub("",e.id))
            reentrant = False
            if len(parent_node_lookup[id]) > 1:
                reentrant = True

            if len(parent_node_lookup[id]) == 0:
                parent_node_idx, argn = 0, 0
                parent_node_lookup[id].append((parent_node_idx + self.n_reentrant, argn))
            #if reentrant:
            #    nidx = self.expr_idx_to_node_idx[eidx]

            op_value_dict = json.loads(e.op.value)
            for i, (parent_node_idx, argn) in enumerate(parent_node_lookup[id]):
                # add type 
                self.node_name_list.append(op_value_dict['schema'])
                # add node idx
                self.node_idx_list.append(nidx)
                # add edge 
                self.edge_head_list.append(parent_node_idx)

                try:
                    val_argn = get_arg_num(id)
                except IndexError:
                    val_argn = 0

                self.edge_type_list.append(f"val_arg-{val_argn}")
                # update lookup 
                self.node_idx_to_expr_idx[nidx] = eidx
                self.expr_idx_to_node_idx[eidx] = nidx
                # increment counter so that edge indices are correct 
                if i > 0:
                    self.n_reentrant += 1 

            # add underlying 
            nested = False
            underlying = op_value_dict['underlying']
            try:
                value_words = underlying.strip().split(" ")
            except AttributeError:
                try:
                    assert(type(underlying) in [int, float, bool, list])
                except AssertionError:
                    pdb.set_trace() 
                # deal with underlying lists
                if type(underlying) == list:                         
                    if len(underlying) >= 1:                            
                        # nested 
                        nested = True
                        outer_value_words = []
                        for word_or_list in underlying[0][0]:
                            if type(word_or_list) == str:
                                outer_value_words.append(word_or_list)
                            elif type(word_or_list) == list:
                                if type(word_or_list[0]) == list:
                                    outer_value_words.append([str(x) for x in word_or_list[0]])
                                else:
                                    outer_value_words.append([str(x) for x in word_or_list])

                        #outer_value_words = [str(x) for x in underlying[0][0]]
                        try:
                            inner_value_words = [str(x) for x in underlying[1]]
                        except IndexError:
                            inner_value_words = []
                else:
                    value_words = [str(underlying)]

            if not nested:
                for i, word in enumerate(value_words): 
                    self.node_name_list.append(word)
                    self.node_idx_list.append(self.node_idx_list[-1] + 1) 
                    self.edge_head_list.append(nidx + self.n_reentrant)
                    self.edge_type_list.append(f"arg-{i}")
            else:
                # reverse so that the function type comes before the type constraints, e.g. [roleConstraint, [[Constraint, DateTime]], '^']
                outer_value_words = outer_value_words[::-1]
                outer_parent_idx = -1
                for i, word_or_list in enumerate(outer_value_words): 
                    if type(word_or_list) == str:
                        if i == 0:
                            self.node_name_list.append(word_or_list)
                            self.node_idx_list.append(self.node_idx_list[-1] + 1) 
                            self.edge_head_list.append(nidx + self.n_reentrant)
                            self.edge_type_list.append(f"arg-{i}")
                            outer_parent_idx = nidx + i + 1
                        else:
                            assert(word_or_list == "^")
                            # don't add this, just sugar 
                            continue
                    elif type(word_or_list) == list:
                        for j, word in enumerate(word_or_list):
                            self.node_name_list.append(word)
                            self.node_idx_list.append(self.node_idx_list[-1] + 1)
                            # set parent to outer function type 
                            self.edge_head_list.append(outer_parent_idx)
                            self.edge_type_list.append(f"type_arg-{j}")

                for i, word in enumerate(inner_value_words): 
                    self.node_name_list.append(word)
                    self.node_idx_list.append(self.node_idx_list[-1] + 1) 
                    self.edge_head_list.append(outer_parent_idx)
                    self.edge_type_list.append(f"inner_arg-{i}")

        def add_call_like_op(e: Expression, eidx: int, nidx: int):
            id = int(NOBRACK.sub("",e.id))
            try:
                parent_node_idx, argn = parent_node_lookup[id][-1]
            except IndexError:
                # attach to root 
                parent_node_lookup[id] = [(0,0)]

            for i, (parent_node_idx, argn) in enumerate(parent_node_lookup[id]):
                # add type 
                self.node_name_list.append(e.op.name) 
                # add node idx
                self.node_idx_list.append(nidx)
                # add edge 
                self.edge_head_list.append(parent_node_idx)

                try:
                    call_argn = get_arg_num(id)
                except IndexError:
                    call_argn = 0
                    
                self.edge_type_list.append(f"call_arg-{call_argn}")
                # update lookup 
                self.node_idx_to_expr_idx[nidx] = eidx
                self.expr_idx_to_node_idx[eidx] = nidx
                # increment counter so that edge indices are correct 
                if i > 0:
                    self.n_reentrant += 1

            # add type args
            type_args = e.type_args
            if type_args is not None:
                for i, ta in enumerate(type_args): 
                    add_type_name(ta, nidx + self.n_reentrant, i)

            for i, dep in enumerate( e.arg_ids):
                field_node_idx = nidx 
                arg_id = int(NOBRACK.sub("", dep))
                parent_node_lookup[arg_id].append((field_node_idx + self.n_reentrant, i))

        # add dummy root node 
        self.node_name_list.append("@ROOT@")
        self.edge_head_list.append(0)
        self.edge_type_list.append('root')
        self.node_idx_list.append(0)

        # reverse to get correct order 
        program_new.expressions.reverse() 
        for eidx, expr in enumerate(program_new.expressions):
            try:
                node_idx = self.node_idx_list[-1] + 1
            except IndexError:
                # 0 is reserved for root 
                node_idx = 1
            expr_idx = int(NOBRACK.sub("", expr.id))
            if isinstance(expr.op, BuildStructOp):
                add_struct_op(expr, expr_idx, node_idx)
            elif isinstance(expr.op, ValueOp):
                add_value_op(expr, expr_idx, node_idx)
            elif isinstance(expr.op, CallLikeOp):
                add_call_like_op(expr, expr_idx, node_idx)
            else:
                raise ValueError(f"Unexpected Expression: {expr}")

    @staticmethod
    def prediction_to_program(node_name_list: List[str],
                              node_idx_list: List[int], 
                              edge_head_list: List[int], 
                              edge_type_list: List[str]): 

        graph = CalFlowGraph.lists_to_ast(node_name_list, node_idx_list, edge_head_list, edge_type_list)

        def get_arg_children(op_node):
            outgoing_edges = [e for e in graph.edges if e[0] == op_node and graph.edges[e]['type'].startswith("arg")]
            outgoing_edges = sorted(outgoing_edges, key = lambda e: int(graph.edges[e]['type'].split("-")[1]))
            return [e[1] for e in outgoing_edges]

        def get_inner_arg_children(op_node):
            outgoing_edges = [e for e in graph.edges if e[0] == op_node and graph.edges[e]['type'].startswith("inner_arg")]
            outgoing_edges = sorted(outgoing_edges, key = lambda e: int(graph.edges[e]['type'].split("-")[1]))
            return [e[1] for e in outgoing_edges]

        def get_type_arg_children(op_node):
            outgoing_edges = [e for e in graph.edges if e[0] == op_node and graph.edges[e]['type'].startswith("type_arg")]
            outgoing_edges = sorted(outgoing_edges, key = lambda e: int(graph.edges[e]['type'].split("-")[1]))
            return [e[1] for e in outgoing_edges]

        def get_fxn_children(op_node):
            # ordering = fxn_arg, call_arg, value_arg
            order_augment = {"fxn_arg": 0,
                             "call_arg": 0,
                             "val_arg": 0}

            def get_order_num(edge_type):
                etype, argn = edge_type.split("-")
                argn=int(argn)
                eord = order_augment[etype]
                return eord + argn

            outgoing_edges = [e for e in graph.edges if e[0] == op_node and (graph.edges[e]['type'].startswith("fxn_arg")
                                                                            or graph.edges[e]['type'].startswith("val_arg")
                                                                            or graph.edges[e]['type'].startswith("call_arg"))]

            outgoing_edges = sorted(outgoing_edges, key = lambda e: get_order_num(graph.edges[e]['type']), reverse=False)
            return [e[1] for e in outgoing_edges]

        def get_parent(op_node):
            incoming_edges = [e for e in graph.edges if e[1] == op_node]
            return [e[0] for e in incoming_edges][0]

        def get_type_args(node):
            nested = []
            type_arg_node_idxs = get_type_arg_children(node)
            for ta_node in type_arg_node_idxs:
                name = graph.nodes[ta_node]['node_name']
                #ta_subchildren = get_type_arg_children(ta_node)
                type_args = get_type_args(ta_node)
                if type_args is None:
                    type_args = []
                type_name = TypeName(base=name, type_args = type_args)
                nested.append(type_name)
            if len(nested)>0:
                return nested
            return None

        node_id_to_expr_id = {}
        is_expr = [True if (not et.startswith("arg") \
                            and not et.startswith("inner_arg") \
                            and not et.startswith("type_arg")) else False for et in edge_type_list]
        n_expr = sum(is_expr)

        n_reentrant = 0
        for i, node_idx in enumerate(node_idx_list):
            if i - n_reentrant != node_idx:
                n_reentrant += 1

        n_expr -= n_reentrant

        curr_expr = n_expr
        for i in range(len(edge_type_list)):
            # if it's either a build, value, or call
            if is_expr[i]: 
                node_id = node_idx_list[i]
                if node_id not in node_id_to_expr_id.keys():
                    node_id_to_expr_id[node_id] = curr_expr
                    curr_expr -= 1

        expressions = []
        for node in sorted(graph.nodes):
            op = None
            if graph.nodes[node]['function_type'] is not None: 
                if graph.nodes[node]['function_type'] == "build":
                    empty_base = True
                    children = get_fxn_children(node) 
                    field_children = get_arg_children(node)
                    name = graph.nodes[node]['node_name']

                    if len(field_children) == 0:
                        field_names = [None for i in range(len(children))]
                    else:
                        field_names = [graph.nodes[child]['node_name'] for child in field_children]
                        fxn_children = get_fxn_children(node) 
                        if len(field_names) != len(fxn_children):
                            # add difference 
                            field_names = [None for i in range(len(fxn_children) - len(field_names))] + field_names


                    op = BuildStructOp(op_schema = name, 
                                       op_fields = field_names,
                                       empty_base = empty_base, 
                                       push_go = True)

                elif graph.nodes[node]['function_type'] == "value": 
                    children = get_arg_children(node)
                    inner_gchildren = []
                    for n in children:
                        inner_gchildren += get_inner_arg_children(n) 
                    name = graph.nodes[node]['node_name']
                    child_names = [graph.nodes[child]['node_name'] for child in children]
                    underlying = " ".join(child_names) 

                    ## check for ints 
                    if len(child_names) == 1:
                        if child_names[0].isdigit():
                            underlying = int(child_names[0])
                        elif re.match("-?\d+\.\d+", child_names[0]) is not None: 
                            underlying = float(child_names[0])
                        elif child_names[0].strip() in ["True", "False"]:
                            underlying = True if child_names[0].strip() == "True" else False
                        elif child_names[0].strip() == "[]":
                            underlying = []
                        else:
                            # needs to not have spaces for sugaring to work 
                            parent_name = graph.nodes[node]['node_name']
                            grandparent_idx = get_parent(node)
                            grandparent_name = graph.nodes[grandparent_idx]['node_name']
                            if parent_name == "Path" and grandparent_name == "get": 
                                underlying = underlying.strip()

                        # exception: LocationKeyphrases are always str                        
                        if name == "String" and type(underlying) == int: 
                            underlying =  " ".join(child_names) 

                    inner_dict = {"schema": name, "underlying": underlying}
                    op = ValueOp(value=json.dumps(inner_dict))

                elif graph.nodes[node]['function_type'] == "call": 
                    name = graph.nodes[node]['node_name']
                    op = CallLikeOp(name=name)
            else:
                continue

            if op is not None:
                fxn_children = get_fxn_children(node) 
                eid = node_id_to_expr_id[node]
                expr_ids = [node_id_to_expr_id[n] for n in fxn_children]
                children_ids = [f"[{e}]" for e in expr_ids]
                type_args = get_type_args(node)

                curr_expr = Expression(id=f"[{eid}]", op = op, arg_ids = children_ids, type_args = type_args)
                expressions.append(curr_expr)

        expressions.reverse()
        return Program(expressions)

    @staticmethod
    def lists_to_ast(node_name_list: List[str], 
                     node_idx_list: List[str],
                     edge_head_list: List[int], 
                     edge_type_list: List[int]) -> List[Any]:
        """
        convert predicted lists back to an AST 
        """
        # use digraph to store data and then convert 
        graph = nx.DiGraph() 

        # start with 1-to-1 mapping 
        node_idx_to_list_idx_mapping = {k:k for k in range(len(node_name_list))}

        def update_mapping_after_n(n):
            for k in node_idx_to_list_idx_mapping.keys():
                if k > n:
                    node_idx_to_list_idx_mapping[k] -= 1

        offset = 0
        for i, (node_name, node_idx, edge_head, edge_type) in enumerate(zip(node_name_list, node_idx_list, edge_head_list, edge_type_list)):
            if edge_type.startswith("fxn_arg"):
                function_type = "build"
            elif edge_type.startswith("val_arg"):
                function_type = "value"
            elif edge_type.startswith("call_arg"):
                function_type = "call"
            else:
                function_type = None 

            reentrant = False
            if i - offset != node_idx:
                # reentrant 
                curr_name = graph.nodes[node_idx]['node_name']
                # check we're not renaming anything here 
                try:
                    assert(curr_name == node_name)
                except AssertionError:
                    #pdb.set_trace() 
                    pass
                offset += 1
                update_mapping_after_n(node_idx) 

            graph.add_node(node_idx, node_name = node_name, function_type= function_type)

            # root self-edges
            if edge_head < 0:
                edge_head = 0

            edge_head_idx = node_idx_list[edge_head] 
            graph.add_edge(edge_head_idx, node_idx, type=edge_type)

        return graph 

    def get_list_data(self, 
                     bos: str = None, 
                     eos: str = None, 
                     bert_tokenizer = None, 
                     max_tgt_length: int = None):
        """
        Converts SMCalFlow graph into a linearized list of tokens, indices, edges, and affiliated metadata 

        """

        node_name_list = self.node_name_list
        tgt_tokens = self.node_name_list
        tgt_indices = self.node_idx_list
        head_indices = self.edge_head_list
        head_tags = self.edge_type_list
        mask = [1 for __ in range(len(tgt_tokens))]

        node_to_idx = defaultdict(list)
        for node_idx, node_id in enumerate(self.node_idx_list):
            node_to_idx[node_id].append(node_idx)

        def trim_very_long_tgt_tokens(tgt_tokens, 
                                    head_tags, 
                                    head_indices, 
                                    mask, 
                                    node_to_idx,
                                    node_name_list):

            tgt_tokens = tgt_tokens[:max_tgt_length]

            head_tags = head_tags[:max_tgt_length]
            head_indices = head_indices[:max_tgt_length]
            mask = mask[:max_tgt_length]



            node_name_list = node_name_list[:max_tgt_length ]

            for node, indices in node_to_idx.items():
                invalid_indices = [index for index in indices if index >= max_tgt_length]
                for index in invalid_indices:
                    indices.remove(index)
            return (tgt_tokens, 
                   head_tags, 
                   head_indices, 
                   mask, 
                   node_to_idx, 
                   node_name_list)

        if max_tgt_length is not None:
            (tgt_tokens, 
             head_tags, 
             head_indices,    
             mask,
             node_to_idx,
             node_name_list) = trim_very_long_tgt_tokens(tgt_tokens, 
                                                       head_tags, 
                                                       head_indices, 
                                                       mask,
                                                       node_to_idx,
                                                       node_name_list)

        copy_offset = 0

        if bos:
            tgt_tokens = [bos] + tgt_tokens
            node_name_list = ["@start@"] + node_name_list
            copy_offset += 1
        if eos:
            tgt_tokens = tgt_tokens + [eos]

        # Target side Coreference
        tgt_token_counter = Counter(tgt_tokens)
        tgt_copy_mask = [0] * len(tgt_tokens)
        for i, token in enumerate(tgt_tokens):
            if tgt_token_counter[token] > 1:
                tgt_copy_mask[i] = 1

        tgt_indices = [i for i in range(len(tgt_tokens))]

        for node, indices in node_to_idx.items():
            if len(indices) > 1:
                copy_idx = indices[0] + copy_offset
                for token_idx in indices[1:]:
                    tgt_indices[token_idx + copy_offset] = copy_idx

        tgt_copy_map = [(token_idx, copy_idx) for token_idx, copy_idx in enumerate(tgt_indices)]
        tgt_copy_indices = tgt_indices[:]

        for i, copy_index in enumerate(tgt_copy_indices):
             # Set the coreferred target to 0 if no coref is available.
             if i == copy_index:
                 tgt_copy_indices[i] = 0

        prev_utt = None
        prev_program = None 
        agent_utt = None
        user_utt = None

        if self.use_context: 
            src_string_split = re.split("(__User)|(__StartOfProgram)|(__Agent)", self.src_str) 
            src_string_split = [x for x in src_string_split if x != '' and x is not None]
            # just 2 cases
            # Utterance
            # Utterance, Program, Utterance 
            if len(src_string_split) == 3:
                user_utt = src_string_split[1].strip()
            elif len(src_string_split) == 7: 
                prev_utt = src_string_split[1].strip()
                if self.use_program:
                    prev_program = src_string_split[3].strip()
                elif self.use_agent_utterance:
                    agent_utt = src_string_split[3].strip() 
                user_utt = src_string_split[5].strip()
            else:
                raise ValueError(f"Expected source string to have one or two turns, got {self.src_str}")
        else:
            user_utts = re.split("__User", self.src_str)
            last_user_utt = user_utts[-1]
            user_utt = re.sub(f"({PROGRAM_SEP}).*", "", last_user_utt) 
            user_utt = re.sub("(__User).*$", "", user_utt) 
            user_utt = user_utt.strip() 
            
        user_utt = user_utt.split(" ")  
        
        # prepend agent utterance 
        if agent_utt is not None:
            agent_utt = agent_utt.split(" ")
            user_utt = ["__Agent"] + agent_utt + ["__User"] + user_utt
        else:
            user_utt = ["__User"] + user_utt

        # prepend context utterance to user utterance 
        if prev_utt is not None:
            prev_utt = prev_utt.split(" ")
            user_utt = ["__User"] + prev_utt + user_utt

        if bert_tokenizer is not None:
            bert_tokenizer_ret = bert_tokenizer.tokenize(user_utt, True)
            src_token_ids = bert_tokenizer_ret["token_ids"]
            src_token_subword_index = bert_tokenizer_ret["token_recovery_matrix"]
        else:
            src_token_ids = None
            src_token_subword_index = None

        src_tokens = user_utt
        original_token_len = len(src_tokens)

        if self.use_program: 
            if prev_program is not None:
                # convert program to sequence of nodes and heads  
                split_prog = prev_program.split(" ")
                # get rid of entities at end 
                while split_prog[-1] != ")":
                    split_prog = split_prog[0:-1]
                    prev_program = " ".join(split_prog)

                prev_graph = CalFlowGraph(src_str="",tgt_str = prev_program)
                program_tokens = prev_graph.node_name_list
                program_inds = [str(x) for x in prev_graph.node_idx_list]
                program_heads = [str(x) for x in prev_graph.edge_head_list]
                program_types = prev_graph.edge_type_list
                # add program nodes as source tokens and add in special token 
                src_tokens = src_tokens + [PROGRAM_SEP] + program_tokens
                # add these things in as source factors 
                src_indices = ["-1" for i in range(original_token_len)] + ["-1"] + program_inds
                src_edge_heads = ["-1" for i in range(original_token_len)] + ["-1"] + program_heads
                src_edge_types = [PAD_EDGE for i in range(original_token_len)] + [PAD_EDGE] + program_types 
            else:
                src_tokens = src_tokens 
                src_indices = ["-1" for i in range(original_token_len)] 
                src_edge_heads = ["-1" for i in range(original_token_len)] 
                src_edge_types = [PAD_EDGE for i in range(original_token_len)] 
        else:
            src_indices = None
            src_edge_heads = None
            src_edge_types = None

        # TODO (elias): we could add some string normalization here to 
        # allow copies without punctuation, e.g. "Darby" in 
        # output can be copied from Darby in input and quotations 
        # can be deterministically added
        src_copy_vocab = SourceCopyVocabulary(src_tokens)
        src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
        src_copy_map = src_copy_vocab.get_copy_map(src_tokens)
        src_must_copy_tags = [0 for t in src_tokens]

        tgt_tokens_to_generate = tgt_tokens[:]
        node_indices = tgt_indices[:]
        if bos:
            node_indices = node_indices[1:]
        if eos:
            node_indices = node_indices[:-1]
        node_mask = np.array([1] * len(node_indices), dtype='uint8')
        edge_mask = np.zeros((len(node_indices), len(node_indices)), dtype='uint8')
        for i in range(1, len(node_indices)):
            for j in range(i):
                if node_indices[i] != node_indices[j]:
                    edge_mask[i, j] = 1 

        # tgt_tokens_to_generate
        tgt_tokens_to_generate = tgt_tokens[:]
        # Replace all tokens that can be copied with OOV.
        for i, index in enumerate(src_copy_indices):
            if index != src_copy_vocab.token_to_idx[src_copy_vocab.unk_token]:
                tgt_tokens_to_generate[i] = DEFAULT_OOV_TOKEN

        for i, index in enumerate(tgt_copy_indices):
            if index != 0:
                tgt_tokens_to_generate[i] = DEFAULT_OOV_TOKEN

        # Bug fix 1: increase by 1 everything, set first to sentinel 0 tok 
        head_indices = [x + 1 for x in head_indices]
        head_indices[0] = 0

        if self.fxn_of_interest is not None and self.fxn_of_interest in tgt_tokens:
            contains_fxn = [1]

        else:
            contains_fxn = [0]

        return {
            "tgt_tokens" : tgt_tokens,
            "tgt_indices": tgt_indices,
            "tgt_copy_indices" : tgt_copy_indices,
            "tgt_copy_map" : tgt_copy_map,
            "tgt_tokens_to_generate": tgt_tokens_to_generate, 
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "tgt_copy_mask" : tgt_copy_mask,
            "src_tokens" : src_tokens,
            "src_token_ids" : src_token_ids,
            "src_token_subword_index" : src_token_subword_index,
            "src_must_copy_tags" : src_must_copy_tags,
            "src_copy_vocab" : src_copy_vocab,
            "src_copy_indices" : src_copy_indices,
            "src_copy_map" : src_copy_map,
            "src_indices": src_indices,
            "src_edge_heads": src_edge_heads,
            "src_edge_types": src_edge_types,
            "calflow_graph": self,
            "node_name_list": node_name_list,
            "contains_fxn": contains_fxn,
            "line_idx": self.line_idx
        }

    @staticmethod
    def prediction_to_string(node_name_list: List[str],
                             node_idx_list: List[int], 
                             edge_head_list: List[int], 
                             edge_type_list: List[int]) -> str:
        program = CalFlowGraph.prediction_to_program(node_name_list, node_idx_list, edge_head_list, edge_type_list) 
        pred_lispress = program_to_lispress(program)
        pred_lispress_str = render_pretty(pred_lispress)
        return pred_lispress_str

    @classmethod
    def from_prediction(cls, 
                        src_str, 
                        node_name_list: List[str], 
                        node_idx_list: List[int], 
                        edge_head_list: List[int], 
                        edge_type_list: List[int]):

        # trim 
        N = len(node_name_list)
        edge_head_list = np.array(edge_head_list[0:N])
        edge_type_list = edge_type_list[0:N]

        edge_head_list = [x-1 for x in edge_head_list]

        try: 
            edge_head_list[0] = 0
        except IndexError:
            return CalFlowGraph(src_str=src_str,
                                tgt_str="( Error )")


        # deal with coreference 
        copy_offset = 0
        for i, node_idx in enumerate(node_idx_list): 
            # after a copy is made, decrement everything following it by the number of copies 
            node_idx_list[i] -= copy_offset
            node_idx = node_idx_list[i]
            # adjust edges so that heads that pointed to the old node now point to the new one 
            edge_head_list[edge_head_list == node_idx] = node_idx - copy_offset 
            # increment if there are more copies 
            if i - copy_offset != node_idx: 
                # is a copy 
                copy_offset += 1

        try:
            program = CalFlowGraph.prediction_to_program(node_name_list, node_idx_list, edge_head_list, edge_type_list) 
        except:
            return CalFlowGraph(src_str=src_str,
                                tgt_str="( Error )")
        try:
            pred_lispress = program_to_lispress(program)
        except:
            return CalFlowGraph(src_str=src_str,
                                tgt_str="( Error )")

        pred_lispress_str = render_pretty(pred_lispress)

        try:
            pred_graph = CalFlowGraph(src_str = src_str,
                                    tgt_str = pred_lispress_str)
        except:
            return CalFlowGraph(src_str=src_str,
                                tgt_str="( Error )")
        
        return pred_graph