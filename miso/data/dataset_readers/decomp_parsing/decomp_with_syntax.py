# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import json
import sys
import pdb 
import logging
from collections import defaultdict, Counter, namedtuple
from typing import List, Dict
from overrides import overrides

import networkx as nx
import numpy as np
import spacy 

from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN

from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY
from miso.data.dataset_readers.decomp_parsing.utils import is_english_punct
#from miso.data.dataset_readers.decomp_parsing.decomp import (DecompGraph, parse_attributes, WORDSENSE_RE, 
#                                                            QUOTED_RE, NODE_ATTRIBUTES, SPACY_MODEL, 
#                                                            SourceCopyVocabulary)
from miso.data.dataset_readers.decomp_parsing.decomp import (DecompGraph, parse_attributes, WORDSENSE_RE, 
                                                            QUOTED_RE, NODE_ATTRIBUTES, 
                                                            SourceCopyVocabulary)
#from decomp.semantics.uds import UDSSentenceGraph

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# spacy model required for decoding at test time, when POS tags needed for an input string 
#SPACY_MODEL = "en_core_web_sm"
#try:
#  nlp = spacy.load(SPACY_MODEL)
#except:
#  import subprocess
#  logger.info(f"Downloading spacy model: {SPACY_MODEL}")
#  subprocess.check_output(["python", "-m", "spacy", "download", SPACY_MODEL])
#  nlp = spacy.load(SPACY_MODEL)

class DecompGraphWithSyntax(DecompGraph): 

    def __init__(self, graph, keep_punct = False, drop_syntax = True, order = "inorder", syntactic_method = "concat", full_ud_parse = False): 
        """
        :param graph: nx.Digraph
            the input decomp graph from UDSv1.0
        :param keep_punct: bool
            keep punctuation in the graph
        :param drop_syntax: bool
            flag to replace non-head syntactic relations with "nonhead" edge label
        :param order: 'sorted', 'inorder', or 'reversed'
            the linearization order. 'inorder' means that nodes are not sorted and 
            match the true word-order most closely. 'inorder' has highest performance
        """
        # remove non-semantics, non-syntax nodes
        super(DecompGraphWithSyntax, self).__init__(graph, keep_punct, drop_syntax, order) 
        self.syntactic_method = syntactic_method 
        self.full_ud_parse = full_ud_parse

    @overrides 
    def get_list_node(self, semantics_only):
        """
        Does DFS to linearize the decomp graph, this time keeping all syntax-syntax edges and still propagating syntax info to semantics nodes

        Params
        ------
            semantics_only: true if you only want to parse semantics nodes and drop syntax nodes from the output arboresence entirely 
        """
        drop_syntax = True 

        arbor_graph = nx.DiGraph()
        root_id =  self.graph.rootid

        # check if the graph is valid
        if len(self.graph.semantics_subgraph.nodes) == 0 or self.graph_size == 0:
            if not self.full_ud_parse: 
                #print(f"skipping for not having semantics nodes")
                return None, None, None
            else:
                # FOR UD SOTA RESULT: assign a null semantics graph to the datapoint 
                nx_g = self.graph.graph 
                name = self.graph.name 
                syn_node_name = list(self.graph.syntax_subgraph.nodes.keys())[0]
                sem_node_name = re.sub("syntax","semantics-arg", syn_node_name) 
                
                attr_dict = {"domain":"semantics", 'frompredpatt': True, "type": "argument"} 
                nx_g.add_node(sem_node_name, **attr_dict) 
                edge_dict = {'domain': 'interface', 'type': 'head', 'id': syn_node_name}
                nx_g.add_edge(sem_node_name, syn_node_name, **edge_dict) 
                # overwrite  
                self.graph = UDSSentenceGraph(nx_g, name) 
                self.remove_performative(self.graph) 

        # adding this because you can visit a node as many times as it has incoming edges 
        visitation_limit = {}
        synt = self.graph.syntax_subgraph

        # add node ids to synatx nodes
        attrs = {}
        for node in synt.nodes:
            try:
                form = synt.nodes[node]['form']
            except KeyError:
                form = ""
            try:
                upos = synt.nodes[node]['upos']
            except KeyError:
                upos = ""
            attrs[node] = {"form": form,
                            "upos": upos,
                            "id": node}

        self.graph.add_annotation(node_attrs = attrs,  edge_attrs = {})
        added_synt = [] 
        # add all nodes in the semantics subgraph to new graph
        # collect all nodes first and label semantics nodes with their syntactic head
        sem_nodes = list(self.graph.semantics_nodes)
        syn_deps = {}

        spans = {}


        def children(node):
            return [e[1] for e in self.graph.graph.edges if e[0] == node]
        
        # deal with embedded preds
        to_add = []
        to_remove = []
        to_remove_nodes = []
        for node_a  in self.graph.nodes:
            if "syntax" in node_a:
                continue
            for node_b in self.graph.nodes:
                if "syntax" in node_b:
                    continue
                if node_a == node_b:
                    continue

                # node a is pred, node b is arg
                if node_b == re.sub("-pred-", "-arg-", node_a):
                    self.graph.nodes[node_b]['text'] = "SOMETHING"

        self.sem_visited = []

        # semantic node dfs to deal with the removal of preformative
        # and reattach the nodes to a root node, and figure out
        # which semantic parent node is closest to a span, which deals
        # with overlapping spans 
        def sem_depth_dfs(node, depth):
            if node not in self.sem_visited:
                try:
                    span = set([x  for l in self.graph.span(node, attrs = ['id']).values() for x in l]) - set([root_id])
                except (ValueError, KeyError):
                    span = set([])
                spans[node] = (depth, span)
                self.sem_visited += [node]

                for c in children(node):
                    sem_depth_dfs(c, depth+1) 
   
        for root in self.sem_roots:
            sem_depth_dfs(root, 1)

        # assign the syntactic span to the nearest dominating
        # syntax node
        for n1, (d1, span1) in spans.items():
            for n2, (d2, span2) in spans.items():
                if n1 == n2:
                    continue 
                else:
                    if (d1 < d2 and len(span1 & span2) > 0):
                        no_intersect = span1 - span2
                        spans[n1] = (d1, no_intersect)

        # find the syntactic head, resorting to replacing "semantics"
        # with "syntax" where necessary (when error in UDSv1.0 graph) , rare
        for node in sorted(self.graph.semantics_subgraph.nodes):
            if node in self.ignore_list:
                continue
            try:
                syn_dep = self.graph.head(node, attrs = ["form", "upos", 'id'])
            except (IndexError, KeyError) as e:
                # add syntactic dependency head manually 
                synt_node = re.sub("semantics", "syntax", node)
                synt_node = re.sub("-arg", "", synt_node)
                synt_node = re.sub("-pred", "", synt_node)
                num = int(synt_node.split("-")[2])
                synt_d = self.graph.nodes[synt_node]
                syn_dep = (num, [synt_d['form'], synt_d['upos'], synt_d['id']])


            syn_head_id = syn_dep[1][2]
            try:
                arbor_graph.add_node(node, text = syn_dep[1][0], pos = syn_dep[1][1], 
                                    **{k:v for k,v in self.graph.nodes[node].items() 
                                    if k not in ['form', 'upos', 'id']})
            except TypeError:
                # already added SOMETHING text
                arbor_graph.add_node(node, pos = "@@UNK@@", 
                                    **{k:v for k,v in self.graph.nodes[node].items() 
                                    if k not in ['form', 'upos', 'id']})

            added_synt.append(syn_head_id)

            incoming = [se for se in self.graph.semantics_edges(node) if se[1] == node]

            visitation_limit[node] = len(incoming)
            syn_deps[node] = syn_dep

            # add the other syntactic children of the sem node
            if not semantics_only:
                # only add semantics children if we're not training strictly on semantics nodes
                try:
                    __, syn_children_ids = spans[node]
                    syn_children = {i:[self.graph.nodes[c]['form'], self.graph.nodes[c]['upos'], self.graph.nodes[c]['id']] for i, c in enumerate(syn_children_ids)}

                except KeyError:
                    # copula
                    assert('semantics' in node and 'arg' in node)
                    syn_children = {}
                
                for (idx, (text, pos, syn_child)) in syn_children.items():
                    if syn_child not in added_synt:
                        arbor_graph.add_node(syn_child, text = text, pos = pos,
                                                **{k:v for k,v in self.graph.nodes[syn_child].items() 
                                                if k not in ['form', 'upos', 'id']})

                        if drop_syntax:
                            edge_label = 'nonhead'
                        else:
                            edge = (syn_head_id, syn_child)
                            try:
                                edge_label = self.graph.edges[edge]['type']
                            except KeyError:
                                # sometimes it's not in the graph
                                edge_label = "nonhead"

                        arbor_graph.add_edge(node, syn_child, semrel = edge_label)

                        visitation_limit[syn_child] = 1
                        added_synt.append(syn_child)

        if not semantics_only:
            ids_by_node = {}
            for node in  sorted(self.graph.semantics_subgraph.nodes):
                if node in self.ignore_list:
                    continue
                syn_dep = syn_deps[node]
                syn_head_id = syn_dep[1][2]
                # now collect the span dominated by the syntactic head
                # ids keeps track locally of what is dominated by syn_head_id
                # added_synt keeps track globally of which syntax nodes have been added to graph so that we don't double-add
                # depth keeps track of how distant syn node is from root so that if it's headed by two we can assign to closer
                ids = []
                def syn_dfs(top_node, depth):
                    if top_node not in ids: 
                        syn_children = [e[1] for e in self.graph.edges(top_node) if "syntax" in e[1]]
                        for child in syn_children:
                            if not self.keep_punct: 
                                if self.graph.nodes[child]['upos'].lower() not in  ['punct']:
                                    syn_dfs(child, depth + 1)
                                    if child not in added_synt:
                                        ids.append((child, depth))
                            else:
                                syn_dfs(child, depth + 1)
                                if child not in added_synt:
                                    ids.append((child,depth))

                syn_dfs(syn_head_id, 0)
                # remove syntactic head so that it's not doubled
                ids = [x for x in ids if x not in added_synt]
                ids_by_node[node] = ids 
                # expand ids to include all syntactic children
                # add all syntax nodes under the semantics node up to leaves
                nodes_to_add = []
            
            # postprocess ids to resolve cases where a syntactic node 
            # is dominated by two different nodes and we have to pick the closer one
            for node_a, ids_and_depths_a in ids_by_node.items():
                for node_b, ids_and_depths_b in ids_by_node.items():
                    if node_a == node_b:
                        continue
                    for a_idx, (id_a, d_a) in enumerate(ids_and_depths_a):
                        for b_idx, (id_b, d_b) in enumerate(ids_and_depths_b):
                            if id_a == id_b:
                                # same syn node headed by two different nodes 
                                # take less depth
                                if d_a < d_b:
                                    # delete b
                                    popped = ids_by_node[node_b].pop(b_idx) 
                                else:
                                    # delete a
                                    popped = ids_by_node[node_a].pop(a_idx) 
            

            already_dominated = []

            for node in  sorted(self.graph.semantics_subgraph.nodes):
                if node in self.ignore_list:
                    continue
                ids = ids_by_node[node] 
                for i, (syn_node_id, __) in enumerate(ids):
                    visitation_limit[syn_node_id] = 1
                    nodes_to_add.append((syn_node_id, 
                                        self.graph.nodes[syn_node_id]['form'], 
                                        self.graph.nodes[syn_node_id]['upos']))


                for (syn_node_id, text, pos) in nodes_to_add:
                        if syn_node_id not in added_synt:
                            arbor_graph.add_node(syn_node_id, 
                                                text = text,
                                                pos = pos)

                            added_synt.append(syn_node_id)

                        if drop_syntax:
                            edge_label = 'nonhead'
                        else:
                            edge = (syn_head_id, syn_node_id)
                            try:
                                edge_label = self.graph.edges[edge]['type']
                            except KeyError:
                                # sometimes it's not in the graph
                                edge_label = "nonhead"

                        if syn_node_id not in already_dominated: 
                            arbor_graph.add_edge(node, syn_node_id, semrel = edge_label)
                            already_dominated.append(syn_node_id)

        # copy semantics edges
        for e in self.graph.semantics_subgraph.edges:
            if e[0] in self.ignore_list or e[1] in self.ignore_list:
                continue

            e_val = self.graph.semantics_subgraph.edges[e] 
            if e[0] != e[1]:
                e_val['semrel'] = e_val['type']
                arbor_graph.add_edge(*e, **e_val)
        
        visited = defaultdict(int)
        node_list = []


        # get root, the only node that has nothing incoming
        all_sources = [e[0] for e in arbor_graph.edges]
        all_targets = [e[1] for e in arbor_graph.edges]
        potential_roots = [x for x in arbor_graph.nodes if x in all_sources and x not in all_targets]

        # add dummy root
        semantic_root = "dummy-semantics-root"
        visitation_limit[semantic_root] = 1
        arbor_graph.add_node(semantic_root, domain='semantics')
        for pot_root in potential_roots:
            #arbor_graph.add_edge(semantic_root, pot_root, semrel = "root")
            arbor_graph.add_edge(semantic_root, pot_root, semrel = "dependency")

        def dfs(node, relations, parent):
            if visited[node] <= visitation_limit[node]:
                node_list.append((node, relations, parent))
                # haven't visited, visit children
                visited[node] += 1
                if self.order not in ["sorted", "inorder", "reverse"]:
                    logger.warn(f"Invalid sorting order: {self.order}")
                    logger.warn(f"Reverting to 'inorder' ordering") 
                    self.order = "inorder"

                if self.order == "sorted":
                    child_edges = sorted([e for e in arbor_graph.edges if e[0] == node] )

                elif self.order == "inorder":
                    # inorder is the best sorting order and corresponds most closely
                    # to the order of the words in the text 
                    child_edges = [e for e in arbor_graph.edges if e[0] == node]
                    sem_edges = [e for e in child_edges if "semantics" in e[0] and "semantics" in e[1]]
                    syn_edges = sorted([e for e in child_edges if "syntax" in e[0] or "syntax" in e[1]], key = lambda x:int(x[1].split("-")[-1]))
                    
                    child_edges = sem_edges + syn_edges

                elif self.order == "reverse":
                    child_edges = sorted([e for e in arbor_graph.edges if e[0] == node], reverse=True)

                else:
                    pass

                parent = node

                # linearize 
                for child_e in child_edges:
                    relations = {k:v for k,v in arbor_graph.edges[child_e].items() if k not in ["domain", "type", "frompredpatt"]}
                    child = child_e[1]
                    dfs(child, relations, parent)

        dfs(semantic_root,
            {'semrel': 'dependency'},
            semantic_root)
            
        # set arbor graph for eval later
        self.arbor_graph = arbor_graph

        return node_list, [semantic_root] , arbor_graph


    def linearize_syntactic_graph(self):
        """ 
        do BFS on the syntax graph to get 
        a list of nodes, head indices, edge labels 
        doesn't add EOS or BOS tokens since those 
        change depending on concat/combo strategy 
        """
        node_list = []
        node_name_list = []
        head_inds = []
        head_labels = []
        head_mask = []  

        syntax_graph = self.graph.syntax_subgraph
        if len(syntax_graph.nodes) == 0:
            # test-time, lenght is 0
            return node_list, node_name_list, head_inds, head_labels, head_mask 

        possible_roots = set(syntax_graph.nodes.keys())
        for source_node, target_node in syntax_graph.edges:
            possible_roots -= set([target_node])

        try:
            assert(len(possible_roots) == 1)
        except AssertionError:
            return [], [], [], [], []

        root = list(possible_roots)[0]

        head_lookup = {root: 0}

        # do BFS
        idx = 0
        frontier = [root]
        while len(frontier) > 0:
            curr_node = frontier.pop(0)
            node_list.append(syntax_graph.nodes[curr_node]['form'])
            node_name_list.append(curr_node) 
           
            head_inds.append(head_lookup[curr_node])
            # get deprel 
            if len(head_inds) > 0:
                head_node = node_name_list[head_inds[-1]]
            else:
                head_node = root 
                
            if head_node == "BOS":
                # root 
                head_node = curr_node

            edge = (head_node, curr_node)
            try:
                label = syntax_graph.edges[edge]['deprel']
            except KeyError:
                # root 
                label = "root" 

            head_labels.append(label)
            head_mask.append(1)
            curr_children = [e[1] for e in syntax_graph.edges if e[0] == curr_node]
            for c in curr_children:
                head_lookup[c] = idx

            frontier += curr_children
            idx += 1

        return node_list, node_name_list, head_inds, head_labels, head_mask 


    @overrides 
    def get_list_data(self, bos=None, eos=None, bert_tokenizer=None, max_tgt_length=None, semantics_only = False):
        """
        convert a decomp graph into a shallow format where semantics nodes are labelled with their syntactic head
        and syntax nodes are simplified to have their semantic parent node as governor. All semantic edges are preserved
        except in cases of embedded predicates, where only the predicate node is preserved and all children of the 
        semantic-arg node supervening on the predicate node in the original graph are assigned to the predicate node,
        simplifying the graph and removing the argument node which does not have a unique span in the surface form. 

        After converting the graph, traverses the new graph (called an arbor_graph) and returns a list of nodes and relations.
        Using this list, builds all the data to be put into fields by ~/data/datset_readers/decomp_reader
        """

        def concat_two(tokens1, tokens2, 
                       heads1, heads2,
                       labels1, labels2,
                       mask1, mask2,
                       names1, names2, 
                       add_bos=True,
                       add_eos=True,
                       add_sep=True):

            offset = len(tokens1) 
            if add_bos:
                tokens1 = ["@start@"] + tokens1
                names1 = ["BOS"] + names1 
                mask1 = [0] + mask1
            if add_eos:
                tokens2 = tokens2 + ["@end@"]
                names2 = names2 + ["EOS"]
                mask2 = mask2 + [0]
            if add_sep: 
                tokens1 = tokens1 + ["@syntax-sep@"]
                heads1 = heads1 + [-2]
                labels1 = labels1 + ["SEP"]
                names1 = names1 + ["SEP"]
                mask1 = mask1 + [0]
                offset += 1
            tokens = tokens1 + tokens2
            # increment heads by offset of first seq 
            heads2 = [x + offset for x in heads2]
            # root is sentinel 
            heads2[0] = -1
            heads = heads1 + heads2
            labels = labels1 + labels2 
            mask = mask1 + mask2
            names = names1 + names2

            return tokens, heads, labels, mask, names 

        node_list, sem_roots, arbor_graph = self.get_list_node(semantics_only)

        if node_list is None:
            return None

        tgt_tokens = []
        tgt_head_tokens = []
        tgt_attributes = []
        edge_attributes = []
        head_indices = []
        head_tags = []
        mask = []
        node_name_list = []

        node_to_idx = defaultdict(list)
        visited = defaultdict(int)

        def flatten_attrs(layered_dict):
            # flatten decomp new structure and get masks
            to_ret = {}
            for outer_key, inner_dict in layered_dict.items():
                try:
                    for inner_key, inner_vals in inner_dict.items():
                        new_key = f"{outer_key}-{inner_key}"
                        to_ret[new_key] = inner_vals
                except (KeyError, AttributeError) as e:
                    to_ret[outer_key] = {"value": inner_dict, "confidence": 1.0}

            return to_ret 

        def update_info(node, relation, parent, token, attrs):
            # if it has semrel, then it's not protoroles
            if 'semrel' in relation.keys() and relation['semrel'] not in ['dependency', 'head']:
                # changing this so that pred_arg mask is on at all locations
                head_indices.append(node_to_idx[parent][-1])
                head_tags.append("EMPTY")
                # changing this so that pred_arg mask is on at all locations
                mask.append(1)
            else:
                head_tags.append(relation['semrel'])
                head_indices.append(node_to_idx[parent][-1])
                mask.append(1)

            edge_attrs = {k:v for k,v in relation.items() if k not in ['semrel', 'id']}
            edge_attributes.append(flatten_attrs(edge_attrs))

            tgt_tokens.append(token)
            node_name_list.append(node)
            tgt_attributes.append(flatten_attrs(attrs))

        for node, relation, parent_node in node_list:
            node_to_idx[node].append(len(tgt_tokens))
            try:
                instance = arbor_graph.nodes[node]['text']
                attrs = {k:v for k,v in arbor_graph.nodes[node].items() if self.re_in(k, NODE_ATTRIBUTES)}
            except KeyError:
                # is root
                instance = "@@ROOT@@"
                attrs = {}

            update_info(node, relation, parent_node, instance, attrs)

            visited[node] = 1


        # add syntactic subgraph 
        (syn_tokens, syn_node_name_list, 
         syn_head_indices, syn_head_tags, 
         syn_mask)  = self.linearize_syntactic_graph()


        def reorder_syntax_for_encoder(tokens, inds, tags, mask, nodes):
            """
            reorder tokens and relabel indices so that order corresponds to syntactic order 
            """
            if len(tokens) == 0:
                # test time, no nodes 
                op_vec = np.zeros((1, len(tokens)+1, 3))
                return tokens, inds, tags, mask, nodes, op_vec

            # nodes has corrected ordering 
            everything_zipped = zip(tokens, inds, tags, mask, nodes)
            correct_order_zipped = sorted(everything_zipped, key = lambda x: int(x[-1].split("-")[-1]))
            new_tokens, new_inds, new_tags, new_mask, new_nodes = [list(x) for x in zip(*correct_order_zipped)]
            # get mapping from old inds to new inds 
            for i, (head_idx) in enumerate(new_inds):
                head_node = nodes[head_idx]
                new_head_idx = new_nodes.index(head_node)
                new_inds[i] = new_head_idx

            # set root to 0 and increment by 1 
            new_inds = [x + 1 for x in new_inds]
            new_inds[new_tags.index("root")] = 0

            # get operation vector 
            # 0: none, 1: left, 2: right
            op_vec = np.zeros((1, len(tokens)+1, 3))
            # add sentinel to front 
            inds_for_op = [0] + [x-1 if x > 0 else 0 for x in new_inds ]
            for i, head_idx in enumerate(inds_for_op): 
                if head_idx < i:
                    op_vec[0, i, 1] = 1
                elif head_idx > i:
                    op_vec[0, i, 2] = 1
                else:
                    op_vec[0, i, 0] = 1

            return new_tokens, new_inds, new_tags, new_mask, new_nodes, op_vec
        
        if not self.syntactic_method.startswith("concat"): 
            # get rid of bos token
            sem_tokens = tgt_tokens[1:]
        else:
            sem_tokens = tgt_tokens

        sem_head_indices = head_indices
        sem_head_tags = head_tags
        sem_mask = mask 
        sem_node_name_list = node_name_list
    
        true_conllu_dict = None 
        op_vec = None 

        if self.syntactic_method == "concat-after":

            (tgt_tokens, 
             head_indices, 
             head_tags, 
             mask,
             node_name_list) = concat_two(sem_tokens, syn_tokens,
                                          sem_head_indices, syn_head_indices,
                                          sem_head_tags, syn_head_tags,
                                          sem_mask, syn_mask,
                                          sem_node_name_list, syn_node_name_list)

            # pad attributes 
            tgt_attributes = [{}] + tgt_attributes
            edge_attributes = [{}] + edge_attributes
            tgt_attributes += [{} for i in range(len(syn_tokens)+2)]
            edge_attributes += [{} for i in range(len(syn_head_indices)+2)]

            # for conllu 
            syn_heads_for_dict = [x + 1 for x in syn_head_indices]
            syn_heads_for_dict[0] = 0

            # need to skip since truncating might mess up the graphs big time
            if len(tgt_tokens) > max_tgt_length:
                return None

        elif self.syntactic_method == "concat-before":
            (tgt_tokens, 
             head_indices, 
             head_tags, 
             mask,
             node_name_list) = concat_two(syn_tokens, sem_tokens,
                                          syn_head_indices, sem_head_indices,
                                          syn_head_tags, sem_head_tags, 
                                          syn_mask, sem_mask,
                                          syn_node_name_list, sem_node_name_list)
            # offset node_to_idx 
            for node, idx_list in node_to_idx.items():
                idx_list = [x + len(syn_tokens) + 1 for x in idx_list]
                node_to_idx[node] = idx_list 

            # pad attributes 
            tgt_attributes = [{} for i in range(len(syn_tokens)+2)] + tgt_attributes + [{}]
            edge_attributes = [{} for i in range(len(syn_head_indices)+2)] + edge_attributes + [{}]

            # for conllu 
            syn_heads_for_dict = [x + 1 for x in syn_head_indices]
            syn_heads_for_dict[0] = 0

            if len(tgt_tokens) > max_tgt_length:
                return None

        elif self.syntactic_method == "concat-just-syntax":
            # add bos and eos 
            #syn_tokens = ["@start@"] + syn_tokens + ["@syntax-sep@"]
            #syn_node_name_list = ["BOS"] + syn_node_name_list + ["EOS"]
            #syn_mask = [0] + syn_mask + [0]

            tgt_tokens = syn_tokens
            head_indices = syn_head_indices
            head_tags = syn_head_tags
            mask = syn_mask
            node_name_list = syn_node_name_list
            tgt_attributes = [{} for i in range(len(syn_tokens))] 
            edge_attributes = [{} for i in range(len(syn_head_indices))] 

            if len(tgt_tokens) > max_tgt_length:
                return None

        elif self.syntactic_method == "encoder-side":
            # add bos, eos to semantics 
            # no bos or eos for syntax, but it needs to re-ordered 
            (syn_tokens, 
            syn_head_indices, 
            syn_head_tags, 
            syn_mask, 
            syn_node_name_list,
            op_vec) = reorder_syntax_for_encoder(syn_tokens,
                                                    syn_head_indices,
                                                    syn_head_tags,
                                                    syn_mask,
                                                    syn_node_name_list)
            # for conllu 
            syn_heads_for_dict = syn_head_indices


        else:
            raise NotImplementedError

        true_conllu_dict = self.build_conllu_dict(syn_tokens, syn_heads_for_dict, syn_head_tags)

        syn_node_mask = np.array([1] * len(syn_tokens), dtype='uint8')
        syn_node_indices = [i+1 for i in range(len(syn_tokens))]
        syn_edge_mask = np.ones((len(syn_tokens), len(syn_tokens)), dtype='uint8')

        # increment by one for the biaffine parser 
        syn_head_indices = [x for x in syn_head_indices]
        #syn_head_indices = [x+1 for x in syn_head_indices]

        #print("TRUE") 
        #print(tgt_tokens)
        #print(head_indices)
        #print(head_tags)
        #sys.exit() 
        #inds = [i for i in range(len(head_tags))]
        #print(list(zip(inds, tgt_tokens[1:-1], head_indices, head_tags)))

        #attrs = [True if len(tgt_attr) > 0 else False for tgt_attr in tgt_attributes]
        #print(list(zip(tgt_tokens[:-1], attrs)))

        #print(tgt_attributes)
        #print(edge_attributes)
        #sys.exit() 
        
        # TODO: modified to add back in the syntax EOS if trimmed 
        def trim_very_long_tgt_tokens(tgt_tokens, 
                                    head_tags, 
                                    head_indices, 
                                    mask, 
                                    tgt_attributes, 
                                    edge_attributes, 
                                    node_to_idx,
                                    node_name_list):

            tgt_tokens = tgt_tokens[:max_tgt_length] 

            head_tags = head_tags[:max_tgt_length] 
            head_indices = head_indices[:max_tgt_length] 
            mask = mask[:max_tgt_length] 


            tgt_attributes = tgt_attributes[:max_tgt_length] 
            edge_attributes = edge_attributes[:max_tgt_length] 

            node_name_list = node_name_list[:max_tgt_length] 

            for node, indices in node_to_idx.items():
                invalid_indices = [index for index in indices if index >= max_tgt_length]
                for index in invalid_indices:
                    indices.remove(index)

            return (tgt_tokens, 
                   head_tags, 
                   head_indices, 
                   mask, 
                   tgt_attributes, 
                   edge_attributes, 
                   node_to_idx, 
                   node_name_list)

        if max_tgt_length is not None:
            (tgt_tokens, 
             head_tags, 
             head_indices,    
             mask,
             tgt_attributes, 
             edge_attributes, 
             node_to_idx,
             node_name_list) = trim_very_long_tgt_tokens(tgt_tokens, 
                                                       head_tags, 
                                                       head_indices, 
                                                       mask,
                                                       tgt_attributes, 
                                                       edge_attributes, 
                                                       node_to_idx,
                                                       node_name_list)

        # add bos and eos to semantics 
        copy_offset = 0
        if self.syntactic_method not in ['concat-before', 'concat-after']: 
            if bos:
                tgt_tokens = [bos] + tgt_tokens
                tgt_attributes = [{}] + tgt_attributes
                edge_attributes = [{}] + edge_attributes
                copy_offset += 1
                node_name_list = ["@start@"] + node_name_list

            if eos: 
                tgt_tokens = tgt_tokens + [eos]
                tgt_attributes = tgt_attributes + [{}]
                edge_attributes = edge_attributes + [{}]
                node_name_list =  node_name_list + ["@end@"]

        if self.syntactic_method == "concat-just-syntax":
            # trim off the semantics nodes 
            try:
                for k,v in node_to_idx.items():
                    v=[v[0]]
                    node_to_idx[k] = v
            except IndexError:
                return None

        if self.syntactic_method in ['concat-before', 'concat-after']:
            copy_offset += 1

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


        def add_source_side_tags_to_target_side(_src_tokens, _src_tags):
            try:
                assert len(_src_tags) == len(_src_tokens)
            except AssertionError:
                tgt_tags = [DEFAULT_OOV_TOKEN for i in range(len(tgt_tokens))]
                tag_lut = {DEFAULT_OOV_TOKEN: DEFAULT_OOV_TOKEN,
                           DEFAULT_PADDING_TOKEN: DEFAULT_OOV_TOKEN}
                return tgt_tags, tag_lut
                
            tag_counter = defaultdict(lambda: defaultdict(int))
            for src_token, src_tag in zip(_src_tokens, _src_tags):
                tag_counter[src_token][src_tag] += 1

            tag_lut = {DEFAULT_OOV_TOKEN: DEFAULT_OOV_TOKEN,
                       DEFAULT_PADDING_TOKEN: DEFAULT_OOV_TOKEN}
            for src_token in set(_src_tokens):
                tag = max(tag_counter[src_token].keys(), key=lambda x: tag_counter[src_token][x])
                tag_lut[src_token] = tag

            tgt_tags = []
            for tgt_token in tgt_tokens:
                #sim_token = find_similar_token(tgt_token, _src_tokens)
                sim_token = None
                if sim_token is not None:
                    index = _src_tokens.index(sim_token)
                    tag = _src_tags[index]
                else:
                    tag = DEFAULT_OOV_TOKEN
                tgt_tags.append(tag)

            return tgt_tags, tag_lut

        # Source Copy
        src_tokens, src_pos_tags, from_lines = self.get_src_tokens()
        src_token_ids = None
        src_token_subword_index = None
        src_copy_vocab = SourceCopyVocabulary(src_tokens)
        src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
        #src_copy_indices = [x if x > 1 else 0 for x in src_copy_indices]
        src_copy_map = src_copy_vocab.get_copy_map(src_tokens)
        if len(src_pos_tags) == 0:
            # happens when predicting from just a sentence
            # use spacy to get a POS tag sequence   
            doc = nlp(" ".join(src_tokens).strip())
            src_tokens = [str(token) for token in doc]
            src_pos_tags = [token.pos_ for token in doc]
            from_lines = True
        if from_lines: 
            # set syntax nodes
            syn_tokens = src_tokens
            syn_node_name_list = [str(i) for i in range(len(src_tokens))]
            syn_node_mask = np.array([1] * len(syn_tokens), dtype='uint8')
            syn_edge_mask = np.ones((len(syn_tokens), len(syn_tokens)), dtype='uint8')

        tgt_pos_tags, pos_tag_lut = add_source_side_tags_to_target_side(src_tokens, src_pos_tags)

        if bert_tokenizer is not None:
            bert_tokenizer_ret = bert_tokenizer.tokenize(src_tokens, True)
            src_token_ids = bert_tokenizer_ret["token_ids"]
            src_token_subword_index = bert_tokenizer_ret["token_recovery_matrix"]

        #src_must_copy_tags = [1 if is_abstract_token(t) else 0 for t in src_tokens]
        src_must_copy_tags = [0 for t in src_tokens]

        src_copy_invalid_ids = set(src_copy_vocab.index_sequence(
            [t for t in src_tokens if is_english_punct(t)]))

        #print(tgt_tokens) 
        node_indices = tgt_indices[:]
        #print(node_indices)
        #print(f"before {list(zip(tgt_tokens, node_indices))}") 

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


        # transduction fix 1: increase by 1 everything, set first to sentinel 0 tok 
        head_indices = [x + 1 for x in head_indices]
        head_indices[0] = 0

        return {
            "tgt_tokens" : tgt_tokens,
            "tgt_indices": tgt_indices,
            "tgt_pos_tags": tgt_pos_tags,
            "tgt_attributes": tgt_attributes,
            "tgt_copy_indices" : tgt_copy_indices,
            "tgt_copy_map" : tgt_copy_map,
            "tgt_tokens_to_generate": tgt_tokens_to_generate, 
            "syn_tokens": syn_tokens, 
            "syn_head_indices": syn_head_indices,
            "syn_head_tags": syn_head_tags,
            "syn_node_name_list": syn_node_name_list,
            "syn_node_mask": syn_node_mask,
            "syn_edge_mask": syn_edge_mask,
            "edge_mask": edge_mask,
            "node_mask": node_mask,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "edge_attributes": edge_attributes,
            "tgt_copy_mask" : tgt_copy_mask,
            "src_tokens" : src_tokens,
            "src_token_ids" : src_token_ids,
            "src_token_subword_index" : src_token_subword_index,
            "src_must_copy_tags" : src_must_copy_tags,
            "src_pos_tags": src_pos_tags,
            "src_copy_vocab" : src_copy_vocab,
            "src_copy_indices" : src_copy_indices,
            "src_copy_map" : src_copy_map,
            "pos_tag_lut": pos_tag_lut,
            "src_copy_invalid_ids" : src_copy_invalid_ids,
            "arbor_graph": arbor_graph,
            "node_name_list": node_name_list,
            "true_conllu_dict": true_conllu_dict,
            "op_vec": op_vec
        }

    @staticmethod
    def build_syn_graph(nodes, edge_heads, edge_labels): 
        """
        build the syntactic graph from a predicted set of nodes, 
        edge heads, and edge labels
        """
        try:
            graph = nx.DiGraph()
            for i, n in enumerate(nodes):
                attr = {"form": n}
                graph.add_node(i, **attr)

            for i, (head, label) in enumerate(zip(edge_heads, edge_labels)):
                if head == 0: 
                    # root node not present, add self edge 
                    edge = (i,i)
                else:
                    edge = (i, head-1)
                    if i not in graph.nodes or head-1 not in graph.nodes:
                        pdb.set_trace() 
                attr = {"deprel": label} 
                graph.add_edge(*edge, **attr)
            return graph 

        except IndexError:
            return None

    @staticmethod
    def build_conllu_dict(nodes, edge_heads, edge_labels, text=None):
        """
        turn the syntactic graph into conllu format
        """
        #print(f"nodes {nodes}") 
        #print(f"edge_heads {edge_heads}") 
        #print(f"edge_labels {edge_labels}") 
        try:
            assert(len(nodes) == len(edge_heads) == len(edge_labels))
        except AssertionError:
            assert(len(nodes) > len(edge_heads) )
            assert(len(edge_heads) == len(edge_labels)) 
            # this is when we've trimmed, add empty predictions here 
            edge_heads += ["-" for i in range(len(nodes)-len(edge_heads))]
            edge_labels += ["-" for i in range(len(nodes)-len(edge_heads))]
        colnames = ["ID", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]

        rows = []
        for i, (node, head, deprel) in enumerate(zip(nodes, edge_heads, edge_labels)):
            row = {"ID": str(i+1), "form": node, "lemma": "_", "upos": "_", "xpos": "_", 
                    "feats": "_", "head": str(head), "deprel": deprel,
                    "deps": "_", "misc": "_"}
            rows.append(row) 
        return rows 

    @staticmethod 
    def build_sem_graph(syntactic_method, nodes, node_attr, corefs,
                        edge_heads, edge_labels, edge_attr):
        """
        build the semantic arbor graph from a predicted output
        """
        graph = nx.DiGraph()
        
        #if syntactic_method == "concat-after": 
        #    for i in range(1, len(edge_heads)):
        #        edge_heads[i] -= 1

        real_node_mapping = {}

        # Steps
        #########################################
        # 1. add all nodes and get node mapping #
        #########################################

        for i, (node, attr, coref) in enumerate(zip(nodes, node_attr, corefs)):
            if int(coref)  != i:
                # node is a copy of a previous node, need to adjust heads, etc. 
                real_node_mapping[i] = int(coref) 
                # don't need to add the node
                continue

            else:
                real_node_mapping[i] = i
        
            node_id = f"predicted-{i}"
            attr['text'] = node
            # by default make everything semantic
            attr['type'] = 'semantics'
            graph.add_node(node_id, **attr)


        #############################################
        # 2. add semantic edges and syntactic edges #
        #############################################

        done_heads = [] 
        assert(len(edge_labels)  == len(edge_heads) == len(edge_attr))
        for i, (label, head, attr) in enumerate(zip(edge_labels, edge_heads, edge_attr)):

            # add the edge between the original coreferrent node and new head, if they're repeated 
            try:
                child_idx = real_node_mapping[i]
                head_idx = real_node_mapping[head]
            except KeyError:
                continue

            child = f"predicted-{child_idx}"
            parent = f"predicted-{head_idx}"

            if child == parent:
                # skip root-root edge
                continue

            if label != "EMPTY":
            # both parent and child are semantics nodes 
                #logger.info(f"semrl is {attr['semrel']}")
                attr['semrel'] = label
                graph.nodes[parent]['type'] = 'semantics'
                graph.nodes[child]['type'] = 'semantics'
            else:
                attr = {'semrel':'nonhead'}
                # don't set parent to syntax! that just makes all semantics nodes with syntactic children syntax nodes fool
                #graph.nodes[parent]['type'] = 'syntax'
                graph.nodes[child]['type'] = 'token'

            graph.add_edge(parent, child, **attr)

        ##########################################################
        # 3. clean up orphan nodes for later checking in scoring #
        ##########################################################

        for i, node in enumerate(graph.nodes):
            try:
                __ = graph.nodes[node]['type'] 
            except KeyError:
                graph.nodes[node]['type'] = "token"

        return graph

    @classmethod
    def from_prediction(cls, output, syntactic_method):
        """
        build an arbor graph from a prediction

        Parameters
        ---------
        output: Dict
            output of decomp predictor
        """
        def split_two(split, end, nodes, heads, tags, corefs,
                      node_attr, edge_attr, 
                      node_mask, edge_mask):

            nodes1, nodes2 = nodes[0:split], nodes[split+1:end]
            heads1, heads2 = heads[0:split], heads[split+1:end]
            tags1, tags2 = tags[0:split], tags[split+1:end]
            node_attr1, node_attr2 = node_attr[0:split], node_attr[split+1:end]
            edge_attr1, edge_attr2 = edge_attr[0:split], edge_attr[split+1:end]
            node_mask1, node_mask2 = node_mask[0:split], node_mask[split+1:end]
            edge_mask1, edge_mask2 = edge_mask[0:split], edge_mask[split+1:end]
            corefs1, corefs2 = corefs[0:split], corefs[split+1:end]

            offset = len(nodes1) 

            heads2 = [x - (offset + 2)  for x in heads2]

            # Hack: need to make sure all of the corefs > 0, no hard
            # constraint for that in the code at test time so we post-
            # process
            corefs2 = [x - (offset + 1) for x in corefs2]
            corefs2 = [x if x >= 0 else 0 for x in corefs2]

            heads2[0] = 0

            return (nodes1, nodes2, heads1, heads2, 
                    tags1, tags2, corefs1, corefs2,
                    node_attr1, node_attr2, 
                    edge_attr1, edge_attr2, 
                    node_mask1, node_mask2,
                    edge_mask1, edge_mask2)  

        nodes = output['nodes']

        corefs = output['node_indices']
        edge_heads = output['edge_heads'] 
        edge_tags = output['edge_types']

        node_attr = output['node_attributes'][0]
        edge_attr = output['edge_attributes']
        node_mask = output['node_attributes_mask'][0]
        edge_mask = output['edge_attributes_mask']

        if syntactic_method in ["concat-before", "concat-after"]:
            if "@syntax-sep@" in nodes:
                # split on syntax starter
                split_point = nodes.index("@syntax-sep@")
                end_point = len(nodes)
            else:
                # can't make a prediction until model has learned this 
                logger.info(f"Can't make a prediction because of lack of syntax-sep") 
                return None, None, None
            try:
                output = split_two(split_point, end_point, nodes, edge_heads,
                                   edge_tags, corefs, node_attr,
                                   edge_attr, node_mask, edge_mask)

            except IndexError:
                # any index error means not enough training 
                return None, None, None

        elif syntactic_method == "concat-just-syntax": 
            syn_nodes = nodes
            syn_heads = edge_heads
            syn_tags = edge_tags
            syn_mask = edge_mask
            sem_nodes = []
            sem_heads = []
            sem_tags = []
            corefs = []
            node_attr = []
            edge_attr = []
            node_mask = []

        elif syntactic_method == "encoder-side":
            sem_nodes = nodes
            syn_nodes = output['syn_nodes']
            N = len(syn_nodes) 
            sem_heads = edge_heads 
            syn_heads = output['syn_edge_heads'][0:N]
            sem_tags = edge_tags 
            syn_tags = output['syn_edge_types'][0:N] 

            sem_heads = [x-1 for x in sem_heads]

            #print(f"sem_nodes {sem_nodes}") 
            #print(f"sem_heads {sem_heads}") 
            #print(f"sem_tags {sem_tags}") 
            #print(f"syn_nodes {syn_nodes}") 
            #print(f"syn_heads {syn_heads}") 
            #print(f"syn_tags {syn_tags}") 
            try:
                sem_heads[0] = 0
            except IndexError:
                pass 

        else:
            raise NotImplementedError

        if syntactic_method == "concat-after":
            # unpack output semantics first 
            (sem_nodes, syn_nodes, sem_heads, syn_heads, sem_tags, syn_tags,
            corefs, __, node_attr, __, edge_attr, __, node_mask, __, 
            edge_mask, __) = output 

            sem_heads = [x-1 for x in sem_heads]
            sem_heads[0] = 0
            
            syn_heads = [x + 1 for x in syn_heads]
            syn_heads[0] = 0

        elif syntactic_method == "concat-before": 
            # unpack output syntax first 
            (syn_nodes, sem_nodes, syn_heads, sem_heads, syn_tags, sem_tags,
             __, corefs, __, node_attr, __, edge_attr, __, node_mask, __,
             edge_mask) = output 

            #sem_heads = [x-1 for x in sem_heads]
            sem_heads[0] = 0

        else:
            # encoder side 
            pass

        #print(f"syntax") 
        #print(syn_nodes)
        #print(syn_heads)
        #print(syn_tags)
        #print(f"semantics") 
        #print(sem_nodes)
        #print(sem_heads)
        #print(sem_tags)
        #print(corefs) 
        #print(node_attr)
        #print(edge_attr) 

        # off by 1 fixed here 
        try:
            node_attr = [parse_attributes(node_attr[i], node_mask[i], NODE_ONTOLOGY) for i in range(len(node_attr))][1:] + [{}]
            edge_attr = [parse_attributes(edge_attr[i], edge_mask[i], EDGE_ONTOLOGY) for i in range(len(edge_attr))]
            
            sem_graph = cls.build_sem_graph(syntactic_method, sem_nodes, 
                                            node_attr, corefs,
                                            sem_heads, sem_tags, edge_attr)
            cls.arbor_graph = sem_graph
        except (TypeError, IndexError) as e: 
            sem_graph = None
            cls.arbor_graph = sem_graph

        # trim 
        syn_n = len(syn_nodes) 
        syn_heads = syn_heads[0:syn_n]
        syn_tags = syn_tags[0:syn_n]

        syn_graph = cls.build_syn_graph(syn_nodes, syn_heads, syn_tags)
        conllu_dict = cls.build_conllu_dict(syn_nodes, syn_heads, syn_tags) 

        return sem_graph, syn_graph, conllu_dict

    @staticmethod
    def get_triples(arbor_graph, 
                    semantics_only = False,
                    drop_syntax = True,
                    threshold = 0.05,
                    include_attribute_scores = False):
        """
        return instance, relation and attribute triples for S-scoring 
        instances are ("instance", node_id, node_value)
        relations are (relation_name, node, parent) 
        attributes are (attribute_name, node, attribute_value)
        """
        instances = []
        relations = []
        attributes = []
        if arbor_graph is None:
            return instances, relations, attributes

        def convert_attrs(attrs):
            to_ret = {}
            for key in attrs.keys():
                if type(attrs[key]) == dict:
                    for subkey in attrs[key].keys():
                        fullkey = key + "-" + subkey
                        if fullkey in NODE_ONTOLOGY or fullkey in EDGE_ONTOLOGY:
                            value = attrs[key][subkey]['value']  * attrs[key][subkey]['confidence'] 
                            to_ret[fullkey] = value
                else:
                    to_ret[key] = attrs[key] 
            return to_ret  

        for i, node in enumerate(arbor_graph.nodes):
            try:
                if semantics_only:
                    if node == "dummy-semantics-root":
                        arbor_graph.nodes[node]['type'] = 'semantics'
                        
                    # take care of true grpah nodes 
                    if 'semantics' in node:
                        arbor_graph.nodes[node]['type'] = 'semantics'

                    # pred graph types are already set at this stage 
                    if 'syntax' in node or arbor_graph.nodes[node]['type'] != 'semantics':
                        continue
            except KeyError:
                raise KeyError(f"the following node is broken: {node}, {arbor_graph.nodes[node]}")

            node_attrs = arbor_graph.nodes[node]
            if "frompredpatt" in node_attrs.keys():
                node_attrs = convert_attrs(node_attrs)

            try:
                if node_attrs['text'] == "@@ROOT@@":
                    node_attrs['text'] = "root"
                inst = ("instance", node, node_attrs['text'])
            except KeyError:
                # root node
                try:
                    assert(node == "dummy-semantics-root")
                    inst = ("instance", node, "root")
                except AssertionError:
                    raise AssertionError(f"the following node is broken {node}, {node_attrs}")

            if include_attribute_scores and \
             ('semantics' in node or ('type' in arbor_graph.nodes[node].keys() and arbor_graph.nodes[node]['type'] == 'semantics')) \
             and node != "dummy-semantics-root" \
             and node_attrs['text'] != "root":
                for key in NODE_ONTOLOGY:
                    try:
                        if abs(node_attrs[key]) > threshold:
                            attr = (key, node, node_attrs[key])
                            attributes.append(attr)
                    except KeyError:
                        pass
            instances.append(inst)

        for edge in arbor_graph.edges:
            # skip self-edges

            if edge[0] == edge[1]:
                continue

            if semantics_only:
                try:
                    if 'syntax' in edge[0] or arbor_graph.nodes[edge[0]]['type'] != 'semantics' or\
                       'syntax' in edge[1] or arbor_graph.nodes[edge[1]]['type'] != 'semantics':
                        continue
                
                except KeyError:
                    raise KeyError(f"the following node is broken: {edge}, {arbor_graph.nodes[edge[0]]}\n" + \
                                    f"{arbor_graph.nodes[edge[1]]}")

            edge_attrs = arbor_graph.edges[edge]
            try:
                rel = edge_attrs['semrel']
                if rel != 'arg' and drop_syntax:
                    rel = "nonhead"
                relation = (rel, edge[1], edge[0])

            except KeyError:
                # semantic edge
                rel = "arg"
                relation = ("arg", edge[1], edge[0])

            if include_attribute_scores and rel == "arg":
                node_name = f"{edge[0]}-{edge[1]}"
                # instances are ("instance", node_id, node_value)
                any_annotated = False

                if "frompredpatt" in edge_attrs.keys():
                    edge_attrs = convert_attrs(edge_attrs)

                for key in EDGE_ONTOLOGY:
                    try:
                        if abs(edge_attrs[key]) > threshold:
                            attr = (key, node_name, edge_attrs[key])
                            attributes.append(attr)
                            any_annotated = True
                    except KeyError:
                        # not annotated
                        pass
                if any_annotated:
                    inst = ("instance", node_name, "arg") 
                    instances.append(inst)

            relations.append(relation)

        return instances, relations, attributes

    @staticmethod
    def arbor_to_uds(arbor_graph, syn_graph, name, sentence):
        def get_pred_arg(edge):
            source_node = arbor_graph.nodes[e[0]]
            target_node = arbor_graph.nodes[e[1]]
            if arbor_graph.edges[e]['semrel'] == "nonhead":
                return "syntax" 
            if source_node['node_type'] == 'root':
                # if source is root, must be predicate
                return "pred"
            if source_node['node_type'] == 'pred':
                # child of pred must be arg
                return "arg"
            if source_node['node_type'] == "arg" and source_node["text"] == "SOMETHING":
                # child of something is arg
                return "pred"
            else:
                return "arg"

        uds_subgraph = nx.DiGraph()
        # first add root  
        for node in arbor_graph.nodes:
            if ("root" in node or \
              arbor_graph.nodes[node]['text'] == "@@ROOT@@" or \
              arbor_graph.nodes[node]['text'] == "root"):

                arbor_graph.nodes[node]['node_type'] = "root"
                arbor_graph.nodes[node]['type'] = "argument"
                arbor_graph.nodes[node]['domain'] = "semantics"
                arbor_graph.nodes[node]['frompredpatt'] = True
                
                uds_subgraph.add_node(node, **arbor_graph.nodes[node])

        # add node types
        for i in range(len(arbor_graph.edges)):
            for e in arbor_graph.edges:
                try:
                    if 'node_type' in arbor_graph.nodes[e[1]].keys():
                        continue
                    arbor_graph.nodes[e[1]]['node_type'] = get_pred_arg(e)
                except KeyError:
                    continue

        type_mapping = {"pred": "predicate", "arg": "argument"}
        name_mapping = {}
        current_forms = []

        #split_sent = tokenizer(sentence) 
        # add syntactic nodes and edges first 
        sent_list = []
        for node in syn_graph.nodes:
            node_idx = int(node)  
            try:
                form = syn_graph.nodes[node]['form']
            except KeyError:
                form = "" 
            sent_list.append(form) 
            dep_name = f"predicted-{node_idx}-syntax"
            uds_subgraph.add_node(dep_name, domain="syntax", form = form, type="token", position = node_idx)  

        sentence = " ".join(sent_list) 
        for dep, head in syn_graph.edges:
            head_name = f"predicted-{head}-syntax"
            dep_name = f"predicted-{dep}-syntax"
            dep_token = syn_graph.nodes[dep]['form']
            if head  > 0: 
                # head is root if idx = 0, don't add 
                try:
                    head_token = syn_graph.nodes[head]['form']
                except KeyError:
                    pdb.set_trace() 
                    head_token = ""
                uds_subgraph.add_edge(dep_name, head_name, deprel=syn_graph.edges[(dep,head)]["deprel"], domain="syntax") 

        for node in arbor_graph.nodes:
            node_type = arbor_graph.nodes[node]['node_type']
            node_class = node_type 
            if node_type in ["pred", "arg"]:
                node_class = "semantics"
            node_name = node + '-' + node_class
            form = arbor_graph.nodes[node]['text']

            if node_type in ['pred', 'arg']:
                arbor_graph.nodes[node]['domain'] = "semantics"
                arbor_graph.nodes[node]['type'] = type_mapping[node_type]
                arbor_graph.nodes[node]['frompredpatt'] = True
                # only add semantics nodes 
            if node_type in ['pred', 'arg', 'root']: 
                uds_subgraph.add_node(node_name, **arbor_graph.nodes[node]) 

            name_mapping[node] = node_name

            # add text as syntactic child 
            if node_type in ["pred", "arg"]:
                try:
                    synt_idx = sentence.split(" ").index(form) 
                    synt_name = f"predicted-{synt_idx}-syntax"
                    if synt_name not in uds_subgraph.nodes: 
                        uds_subgraph.add_node(synt_name, domain="syntax", form = "", type="token", position = len(sentence.split(" ") - 1)  )
                        
                    uds_subgraph.add_edge(node_name, synt_name, type = "head", domain = 'interface')
                except ValueError:
                    continue

        for edge in arbor_graph.edges:
            src_node, tgt_node = edge
            src_node_name, tgt_node_name = name_mapping[src_node], name_mapping[tgt_node]
            edge_name = (src_node_name, tgt_node_name) 

            if uds_subgraph.nodes[src_node_name]['type'] == 'predicate' and uds_subgraph.nodes[tgt_node_name]['type'] == 'argument':
                # pred to arg is dependency 
                edge_type = "dependency"
            elif uds_subgraph.nodes[src_node_name]['type'] == 'argument' and uds_subgraph.nodes[tgt_node_name]['type'] == 'predicate':
                # arg to pred is dependency 
                edge_type = "head"
            else:
                edge_type = "nonhead"

            # check semantics-semantics edges
            if ("semantics" in src_node_name or "root" in src_node_name) and \
                "semantics" in tgt_node_name:
                if  uds_subgraph.nodes[tgt_node_name]['type'] == 'root' or\
                    uds_subgraph.nodes[src_node_name]['type'] == 'root':
                    edge_type = 'dependency'

                arbor_graph.edges[edge]['type'] = edge_type 
                arbor_graph.edges[edge]['domain'] = 'semantics'
            #elif "semantics" in src_node_name and "syntax" in tgt_node_name:
            #    arbor_graph.edges[edge]['domain'] = 'interface'
            #    arbor_graph.edges[edge]['type'] = 'nonhead'

            #    if tgt_node_name not in uds_subgraph.nodes:
            #        pdb.set_trace() 

                # add all other edges, dependency, head, or nonhead
                uds_subgraph.add_edge(*edge_name, **arbor_graph.edges[edge])
        for node in uds_subgraph.nodes:
            try:
                assert("domain" in uds_subgraph.nodes[node].keys())
            except AssertionError:
                print(f"node {node} has no attribute domain")
                pdb.set_trace() 
        for edge in uds_subgraph.edges:
            if "domain" not in uds_subgraph.edges[edge]:
                uds_subgraph.edges[edge]['domain'] = 'syntax'
                uds_subgraph.edges[edge]['type'] = 'nonhead'

        # add syntactic graph
        # add any missing nodes (possible due transduction) 
        #for name, node in syn_graph.nodes.items():
        #    try:
        #        if node['form'] not in current_forms:
        #            synt_name = name + "-syntax"
        #            uds_subgraph.add_node(synt_name, form = node['form'], node_type="syntax", domain="syntax", type="token")
        #    except KeyError:
        #        print(name, node) 



        uds_graph = UDSSentenceGraph(uds_subgraph, name)
        uds_graph.graph.nodes['-root-0']['domain'] = "semantics"
        uds_graph.graph.nodes['-root-0']['frompredpatt'] = False
        uds_graph.graph.nodes['-root-0']['type'] = 'root'
        return uds_graph
