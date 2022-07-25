# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#from decomp import UDSCorpus
#from decomp.semantics.uds import UDSSentenceGraph
import networkx as nx 
import json
import pdb

#class TestUDSCorpus(UDSCorpus):
#    """
#    Wrapper object to create a UDS class which can load new input strings and empy
#    graphs, for use at test time with new inputs 
#    """
#    def __init__(self, graphs):
#        super().__init__(self, [], []) 
#        self._graphs = graphs
#    
#    @classmethod
#    def from_ud_lines(cls, path):
#        with open(path) as f1:
#            lines = f1.readlines()
#
#        graphs = {}
#        for i, line in enumerate(lines):
#            try:
#                sent, tags = line.split("\t")
#            except ValueError:
#                pdb.set_trace()
#            tags=tags.strip()
#            tags = tags.split(",")
#            sentence = sent 
#            empty_graph = nx.DiGraph()
#            empty_graph.add_node(f"-root-0")
#            empty_graph.nodes[f"-root-0"]['type'] = 'root'
#            empty_graph.nodes[f"-root-0"]['domain'] = 'semantics'
#            empty_graph.nodes[f"-root-0"]['frompredpatt'] = False
#            empty_graph.nodes[f"-root-0"]['sentence'] = sentence
#            empty_graph.nodes[f"-root-0"]['pos_tags'] = tags
# 
#            name = f"test_graph_{i}"
#            graph_data = nx.adjacency_data(empty_graph)
#            g = UDSSentenceGraph.from_dict(graph_data, name) 
#            graphs[name] = g
#        
#        return cls(graphs) 
#
#
#    @classmethod
#    def from_lines(cls, path):
#        with open(path) as f1:
#            lines = f1.readlines()
#
#        return cls.from_single_line(lines) 
#       
#    @classmethod
#    def from_single_line(cls, line):
#        def tokenize(sent): 
#            # TODO: real tokenization here 
#            return sent.split(" ") 
#        if type(line)== str:
#            lines = [line]
#        else:
#            lines = line 
#
#        graphs = {}
#        for i, line in enumerate(lines):
#            sentence = line.strip() 
#            empty_graph = nx.DiGraph()
#            empty_graph.add_node(f"test-root-0")
#            empty_graph.nodes[f"test-root-0"]['type'] = 'root'
#            empty_graph.nodes[f"test-root-0"]['domain'] = 'semantics'
#            empty_graph.nodes[f"test-root-0"]['frompredpatt'] = False
#            empty_graph.nodes[f"test-root-0"]['sentence'] = sentence
#            for i, node_name in enumerate(tokenize(sentence)):
#                empty_graph.add_node(f"test-syntax-{i+1}") 
#                empty_graph.nodes[f"test-syntax-{i+1}"]["form"] = node_name
#                empty_graph.nodes[f"test-syntax-{i+1}"]["domain"] = 'syntax'
#                empty_graph.nodes[f"test-syntax-{i+1}"]["type"] = 'token' 
#                empty_graph.nodes[f"test-syntax-{i+1}"]["position"] = i+1
#            name = f"test_graph_{i}"
#            graph_data = nx.adjacency_data(empty_graph)
#            g = UDSSentenceGraph.from_dict(graph_data, name) 
#            for node in g.nodes:
#                if 'type' not in g.nodes[node].keys():
#                    g.nodes[node]['type'] = None
#                if 'domain' not in g.nodes[node].keys(): 
#                    g.nodes[node]['domain'] = 'syntax'
#
#            g.nodes[f"test-root-0"]['type'] = 'root'
#            g.nodes[f"test-root-0"]['sentence'] = sentence 
#            g.nodes[f"test-root-0"]['domain'] = "semantics" 
#        
#            graphs[name] = g
#        
#        return cls(graphs) 
