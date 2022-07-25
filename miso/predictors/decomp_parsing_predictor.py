# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from overrides import overrides
from typing import List, Iterator, Any
import numpy
from contextlib import contextmanager
import json
import logging 
import sys
import pdb 

import torch
import spacy
import allennlp
from allennlp.predictors.predictor import Predictor
from allennlp.data import Instance
from allennlp.common.util import JsonDict

from miso.models.decomp_parser import DecompParser
from miso.models.decomp_syntax_parser import DecompSyntaxParser
from miso.models.decomp_syntax_only_parser import DecompSyntaxOnlyParser
from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax
from miso.data.dataset_readers.decomp_parsing.ontology import NODE_ONTOLOGY, EDGE_ONTOLOGY

logger = logging.getLogger(__name__) 

def sanitize(x: Any) -> Any:  # pylint: disable=invalid-name,too-many-return-statements
    """
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):  # pylint: disable=no-member
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, DecompGraph):
        return x
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (spacy.tokens.Token, allennlp.data.Token)):
        # Tokens get sanitized to just their text.
        return x.text
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, 'to_json'):
        return x.to_json()
    else:
        raise ValueError(f"Cannot sanitize {x} of type {type(x)}. "
                         "If this is your own custom class, add a `to_json(self)` method "
                         "that returns a JSON-like object.")


@Predictor.register("decomp_parsing")
class DecompParsingPredictor(Predictor):

    @overrides
    def load_line(self, line:str) -> JsonDict:
        try:
            return json.loads(line)
        except json.decoder.JSONDecodeError:
            return line 

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        pred_graph = DecompGraph.from_prediction(outputs)

        return pred_graph

    @contextmanager
    def capture_model_internals(self) -> Iterator[dict]:
        """
        Context manager that captures the internal-module outputs of
        this predictor's model. The idea is that you could use it as follows:

        .. code-block:: python

            with predictor.capture_model_internals() as internals:
                outputs = predictor.predict_json(inputs)

            return {**outputs, "model_internals": internals}
        """
        results = {}
        hooks = []

        # First we'll register hooks to add the outputs of each module to the results dict.
        def add_output(idx: int):
            def _add_output(mod, _, outputs):
                results[idx] = {"name": str(mod), "output": sanitize(outputs)}
            return _add_output

        for idx, module in enumerate(self._model.modules()):
            if module != self._model:
                hook = module.register_forward_hook(add_output(idx))
                hooks.append(hook)

        # If you capture the return value of the context manager, you get the results dict.
        yield results

        # And then when you exit the context we remove all the hooks.
        for hook in hooks:
            hook.remove()

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance],
                               oracle: bool = False) -> List[JsonDict]:
        self._model.oracle = oracle 
        outputs = self._model.forward_on_instances(instances)
        if oracle:
            # assign predicted node and edge attributes to the graph 
            node_res_dict = {k: {'true_val_with_node_ids': {},
                                 'true_val_list': [],
                                  'pred_val_with_node_ids': {},
                                 'pred_val_list': [],
                                  'total_n': 0} 
                                        for k in NODE_ONTOLOGY}
            edge_res_dict = {k: {'true_val_with_edge_ids': {},
                                 'true_val_list': [],
                                  'pred_val_with_edge_ids': {}, 
                                 'pred_val_list': [],
                                 'total_n': 0}
                                        for k in EDGE_ONTOLOGY}

            nodes_to_ignore_str  = ["@@ROOT@@", "@start@", "@end@"]
            
            # iterate over instances in batch 
            for instance, output in zip(instances, outputs):
                pred_node_attrs = output['node_attributes'][1:]
                true_node_attrs = instance.fields['target_attributes'].labels[1:-1]
                true_node_mask = instance.fields['target_attributes'].masks[1:-1]
            
                nodes = instance.fields['target_tokens'].tokens[1:]
                node_ids = instance.fields['node_name_list'].metadata[1:-1]
                try:
                    assert(len(node_ids) == len(nodes))           
                except AssertionError:
                    print(node_ids)
                    print(len(node_ids))
                    print(nodes)
                    print(len(nodes))
                    sys.exit()

                # filter 
                true_edge_attrs = instance.fields['edge_attributes'].labels[1:-1]
                true_edge_mask = instance.fields['edge_attributes'].masks[1:-1]

                assert(len(true_edge_attrs) == len(true_edge_mask))

                true_edge_labels = instance.fields['edge_types'].tokens
                pred_edge_attrs = output['edge_attributes'][1:len(true_edge_labels) + 1]
                heads = instance.fields['edge_heads'].labels
                # iterate over tokens in instance 
                for i, (pred_attr, true_attr, true_mask) in enumerate(zip(pred_node_attrs, true_node_attrs, true_node_mask)):
                    # only compute for non-padding, non-root, etc.
                    if str(nodes[i]) in nodes_to_ignore_str:
                        continue
    
                    node_id = node_ids[i]
                    # iterate over the onotology
                    for j, key in enumerate(NODE_ONTOLOGY):
                        p = pred_attr[j]
                        t = true_attr[j]
                        m = true_mask[j] 
                        if m.item() > 0:
                            #print(f"node {nodes[i]} node_id {node_id} key {key} mask {m.item()}")
                            node_res_dict[key]['true_val_with_node_ids'][node_id] = t
                            node_res_dict[key]['true_val_list'].append(t)
                            node_res_dict[key]['pred_val_with_node_ids'][node_id] = p
                            node_res_dict[key]['pred_val_list'].append(p)
                            node_res_dict[key]['total_n'] += 1

                for i, (pred_attr, true_attr, true_mask) in enumerate(zip(pred_edge_attrs, true_edge_attrs, true_edge_mask)):
                    # only compute for semantic nodes
                    if true_edge_labels[i] == 'EMPTY':
                        continue

                    head_idx = heads[i]
                    node_id = node_ids[i]
                    head_id = node_ids[head_idx]
                    edge_id = f"{head_id}-{node_id}"
                    for j, key in enumerate(EDGE_ONTOLOGY):
                        p = pred_attr[j]
                        t = true_attr[j]
                        m = true_mask[j]
                        if m.item() > 0:
                            edge_res_dict[key]['true_val_with_edge_ids'][edge_id] = t
                            edge_res_dict[key]['true_val_list'].append(t)
                            edge_res_dict[key]['pred_val_with_edge_ids'][edge_id] = p
                            edge_res_dict[key]['pred_val_list'].append(p)
                            edge_res_dict[key]['total_n'] += 1
                
            node_res_dict.update(edge_res_dict)
            return node_res_dict

        return sanitize(outputs)

@Predictor.register("decomp_syntax_parsing")
class DecompSyntaxParsingPredictor(DecompParsingPredictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        # function hijacked from parent class to return a decomp arborescence instead of printing a line 
        pred_sem_graph, pred_syn_graph, conllu_graph = DecompGraphWithSyntax.from_prediction(outputs, self._model.syntactic_method) 

        if conllu_graph is not None:
            if self._model.syntactic_method in ['concat-before', 'concat-after']:
                text = " ".join([row["form"] for row in conllu_graph])
                outputs['syn_nodes'] = text.split(" ") 
            else:
                text = " ".join(outputs['syn_nodes']) 
            id = 1

            conllu_str = f"# sent_id = train-s{id}\n" +\
                         f"# text = {text}\n" + \
                         f"# org_sent_id = {id}\n"
            colnames = ["ID", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc"]
            
            n_vals = 0 
            n_rows = len(conllu_graph) 
            for row in conllu_graph:
                vals = [row[cn] for cn in colnames]
                conllu_str += "\t".join(vals) + "\n"
                n_vals = len(vals) 

            #if self._model.syntactic_method not in ['concat-before', 'concat-after']:
            # cases where we had to trim 
            if len(outputs['syn_nodes']) > n_rows:
                c = n_rows
                for node in outputs['syn_nodes'][n_rows:]: 
                    #vals = [str(c+1)] + [node] + ["-" for i in range(n_vals-2)]
                    dummy_row = {"ID": str(c+1), "form": node, "lemma": "-", "upos": "-",
                                "xpos": "-", "feats": "-", "head": str(1), "deprel": "amod", "deps": "-", "misc": "-"}
                    vals = [dummy_row[cn] for cn in colnames]
                    conllu_str += "\t".join(vals) + "\n"
                    c += 1

            conllu_str += '\n' 
        else:
            conllu_str = ""

        return pred_sem_graph, pred_syn_graph, conllu_str

