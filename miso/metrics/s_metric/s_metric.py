# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from typing import List, Dict 
import random
import logging
from tqdm import tqdm
from collections import namedtuple
import pdb 

from miso.metrics.s_metric.candidate_mappings import CandidateMappings
from miso.metrics.s_metric.weight_dict import WeightDict
from miso.metrics.s_metric.bleu import BLEU
from miso.metrics.s_metric import utils
from miso.metrics.s_metric import constants
from miso.metrics.s_metric.repr import Triple, FloatTriple
from miso.data.dataset_readers.decomp_parsing.decomp import DecompGraph
from miso.data.dataset_readers.decomp_parsing.decomp_with_syntax import DecompGraphWithSyntax

logger = logging.getLogger(__name__) 

(NORMAL, TEST1, TEST2) = ("normal", "sanity-check-with-smatch", "PredPatt-test")

compute_args = {"seed": 0,
                "iter_num": 1,
                "compute_instance": True,
                "compute_attribute": True,
                "compute_relation": True,
                "log_level": None,
                "sanity_check": False,
                "mode": NORMAL
                }

ComputeTup = namedtuple("compute_args", sorted(compute_args))
c_args = ComputeTup(**compute_args)


class S(object):
    """
    Encapsulates all SPR scoring routines 
    """
    def __init__(self, instance1, attribute1, relation1,
                 instance2, attribute2, relation2,
                 iter_num=4,
                 compute_instance=True,
                 compute_attribute=True,
                 compute_relation=True,
                 mode=NORMAL):
        self.match_triple_dict = {}
        self.log = utils.get_logging()
        self.compute_instance = compute_instance
        self.compute_attribute = compute_attribute
        self.compute_relation = compute_relation
        self.mode = mode
        self.x_node2id = {x.gov: i for i, x in enumerate(instance1)}
        self.y_node2id = {y.gov: i for i, y in enumerate(instance2)}
        self.candidate_mappings, self.weight_dict = self.compute_pool(
            instance1, attribute1, relation1,
            instance2, attribute2, relation2,
        )

    def instance_match_score(self, x, y):
        norm_x, norm_y = x.dep.lower(), y.dep.lower()
        return int(x.dep == y.dep)

    def attribute_match_score(self, x, y):
        if x.rel == y.rel:
            return self.instance_match_score(x, y)
        return 0

    def relation_match_score(self, x, y):
        return int(x.rel == y.rel)

    def compute_pool(self, instance1, attribute1, relation1,
                     instance2, attribute2, relation2):
        """Compute all possible node mapping candidates and their weights
        (the triple matching number gain resulting from mapping one node in
        representation1 to another node in representation2).
        :param instance1: instance triples of representation1
        :param attribute1: attribute triples of representation1
        :param relation1: relation triples of representation1
        :param instance2: instance triples of representation2
        :param attribute2: attribute triples of representation2
        :param relation2: relation triples of representation2

        :return candidate_mappings: a dictionary whose key is the node in
                representation1, and the corresponding value is a set of
                candidate matching nodes in representation2.
        :return weight_dict: a cunstomized dictionary which contains the
                numbers of matching instance/attribute/relation pairs for
                every pair of node mapping.
        """
        candidate_mappings = CandidateMappings()
        weight_dict = WeightDict()
        for x_id, x in enumerate(instance1):
            x_mappings = set()
            candidate_mappings.append(x_mappings)
            if self.compute_instance:
                for y_id, y in enumerate(instance2):
                    try:
                        score = x.similarity(y)
                    except NotImplementedError:
                        score = self.instance_match_score(x, y)
                    if score == 0:
                        continue
                    x_mappings.add(y_id)
                    weight_dict.add_instance_pair(x_id, y_id, score)

        if self.compute_attribute:
            for x in attribute1:
                for y in attribute2:
                    try:
                        score = x.similarity(y)
                    except NotImplementedError:
                        score = self.attribute_match_score(x, y)
                    if score == 0:
                        continue
                    x_id = self.x_node2id[x.node]
                    y_id = self.y_node2id[y.node]
                    weight_dict.add_attribute_pair(x_id, y_id, score)

        if self.compute_relation:
            for x in relation1:
                for y in relation2:
                    try:
                        score = x.similarity(y)
                    except NotImplementedError:
                        score = self.relation_match_score(x, y)
                    if score == 0:
                        continue

                    x1_id = self.x_node2id[x.gov]
                    x2_id = self.x_node2id[x.dep]
                    y1_id = self.y_node2id[y.gov]
                    y2_id = self.y_node2id[y.dep]

                    candidate_mappings[x1_id].add(y1_id)
                    candidate_mappings[x2_id].add(y2_id)
                    if x1_id == x2_id and y1_id == y2_id:
                        # If relations are circle, only update
                        # circle_pair_weight.
                        # This generally should not happen.
                        weight_dict.add_circle_pair(x1_id, y1_id)
                    else:
                        weight_dict.add_relation_pair((x1_id, y1_id), (x2_id, y2_id), score)

        if self.mode == TEST1:
            candidate_mappings.sort()
        #self.log.info("Candidate mappings:\n" + str(candidate_mappings))
        #self.log.info("Weight dictionary:\n" + str(weight_dict))
        #self.log.info("")
        return candidate_mappings, weight_dict

    def smart_init_mapping(self):
        """Initialize mapping based on the concept mapping (smart
        initialization)
        :return mapping: Initialized node mapping between two representations.
        """
        matched_y_node_set = set()
        mapping = []
        no_eligible_mapping_x_nodes = []
        for x_node, x_mappings in self.candidate_mappings.items():
            if len(x_mappings) == 0:
                # No possible mapping.
                mapping.append(-1)
                continue
            for y_node in x_mappings:
                node_pair = (x_node, y_node)
                # TODO: Modify here for cross-lingual evaluations.
                if self.weight_dict.node_pair_match_score(node_pair) > 0:
                    if y_node not in matched_y_node_set:
                        mapping.append(y_node)
                        matched_y_node_set.add(y_node)
                        break
            else:
                # No eligible mapping is found.
                no_eligible_mapping_x_nodes.append(x_node)
                mapping.append(-1)

        # If no eligible mapping is found, generate a random mapping.
        for x_node in no_eligible_mapping_x_nodes:
            x_mappings = list(self.candidate_mappings[x_node])
            while len(x_mappings) > 0:
                if self.mode == TEST1:
                    # Always choose the first one to generate deterministic results.
                    y_node_idx = 0
                else:
                    # Get a random node index.
                    y_node_idx = random.randint(0, len(x_mappings) - 1)
                y_node = x_mappings[y_node_idx]
                if y_node in matched_y_node_set:
                    x_mappings.pop(y_node_idx)
                else:
                    matched_y_node_set.add(y_node)
                    mapping[x_node] = y_node
                    break
        return mapping

    def random_init_mapping(self):
        """Generate a random node mapping.
        :returns randomly-generated node mapping between two representations.
        """
        matched_y_node_set = set()
        mapping = []
        for x_node, x_mappings in self.candidate_mappings.items():
            if len(x_mappings) == 0:
                # No possible mapping.
                mapping.append(-1)
                continue
            found = False
            x_mappings = list(x_mappings)
            while len(x_mappings) > 0:
                # Get a random node index.
                y_node_idx = random.randint(0, len(x_mappings) - 1)
                y_node = x_mappings[y_node_idx]
                if y_node in matched_y_node_set:
                    x_mappings.pop(y_node_idx)
                else:
                    found = True
                    matched_y_node_set.add(y_node)
                    mapping.append(y_node)
                    break
            if not found:
                mapping.append(-1)
        return mapping

    def compute_match(self, mapping):
        #self.log.info("Computing match for mapping:" + str(mapping))
        mapping_tuple = tuple(mapping)
        if mapping_tuple in self.match_triple_dict:
            #self.log.info("Saved value: " +
            #              str(self.match_triple_dict[mapping_tuple]))
            return self.match_triple_dict[mapping_tuple]
        match_num = 0
        match_num_d = {"attr": 0, "circ":0, "inst": 0, "rel": 0 }

        for x_node, y_node in enumerate(mapping):
            if y_node == -1:
                # No x node maps to this node.
                continue
            cur_node_pair = (x_node, y_node)
            if cur_node_pair not in self.weight_dict:
                continue
            self.log.debug("Node_pair: %s <=> %s" %(x_node, y_node))
            inst_match_num  = self.weight_dict.get_instance_pair_weight(cur_node_pair)
            match_num_d['inst'] += inst_match_num
            match_num += inst_match_num
            attr_match_num = self.weight_dict.get_attribute_pair_weight(cur_node_pair)
            match_num_d['attr'] += attr_match_num
            match_num += attr_match_num
            circ_match_num = self.weight_dict.get_circle_pair_weight(cur_node_pair)
            match_num_d['circ'] += circ_match_num
            match_num += circ_match_num
            rel_match_num = self.weight_dict.get_relation_pair_weight(cur_node_pair, mapping)
            match_num_d['rel'] += rel_match_num
            match_num += rel_match_num


        #self.log.info("Match computing complete, result: " + str(match_num_d) + "\n")
        self.match_triple_dict[mapping_tuple] = match_num
        return match_num

    def move_gain(self, mapping, x_node, y_node, new_y_node, match_num):
        """Compute the triple match number gain from the move operation.
        :param mapping: Current node mapping
        :param x_node
        :param y_node: The original y node that x node maps to.
        :param new_y_node: The new y node that x node maps to.
        :param match_num: The current triple match number.
        """
        old_pair = (x_node, y_node)
        new_pair = (x_node, new_y_node)
        new_mapping = mapping[:]
        new_mapping[x_node] = new_y_node
        new_mapping_tuple = tuple(new_mapping)
        if new_mapping_tuple in self.match_triple_dict:
            return self.match_triple_dict[new_mapping_tuple] - match_num
        gain = 0
        # Add the triple match incurred by new_pair.
        if new_pair in self.weight_dict:
            gain += self.weight_dict.get_instance_pair_weight(new_pair)
            gain += self.weight_dict.get_attribute_pair_weight(new_pair)
            gain += self.weight_dict.get_circle_pair_weight(new_pair)
            node_pair_dict = self.weight_dict.relation_pair_weight.get(new_pair, {})
            for (x_node2, y_node2), weight in node_pair_dict.items():
                if new_mapping[x_node2] == y_node2:
                    gain += weight
        # Deduct the triple match incurred by old_pair.
        if new_pair in self.weight_dict:
            gain -= self.weight_dict.get_instance_pair_weight(old_pair)
            gain -= self.weight_dict.get_attribute_pair_weight(old_pair)
            gain -= self.weight_dict.get_circle_pair_weight(old_pair)
            node_pair_dict = self.weight_dict.relation_pair_weight.get(old_pair, {})
            for (x_node2, y_node2), weight in node_pair_dict.items():
                if new_mapping[x_node2] == y_node2:
                    gain -= weight

        self.match_triple_dict[new_mapping_tuple] = match_num + gain
        return gain

    def swap_gain(self, mapping, x_node1, y_node1, x_node2, y_node2, match_num):
        """Compute the triple match number gain from the swap operation.
        :param mapping: Current node mapping.
        :param x_node1
        :param y_node1: The original y node that x node 1 maps to.
        :param x_node2
        :param y_node2: The y node that x node 2 maps to and will be swapped with y_node1.
        :param match_num: The current triple match number.
        """
        new_mapping = mapping[:]
        new_mapping[x_node1] = y_node2
        new_mapping[x_node2] = y_node1
        new_mapping_tuple = tuple(new_mapping)
        if new_mapping_tuple in self.match_triple_dict:
            return self.match_triple_dict[new_mapping_tuple] - match_num
        gain = 0
        new_pair1 = (x_node1, y_node2)
        new_pair2 = (x_node2, y_node1)
        old_pair1 = (x_node1, y_node1)
        old_pair2 = (x_node2, y_node2)
        if x_node1 > x_node2:
            new_pair1 = (x_node2, y_node1)
            new_pair2 = (x_node1, y_node2)
            old_pair1 = (x_node2, y_node2)
            old_pair2 = (x_node1, y_node1)
        # Add the triple match incurred by new_pair1 and new_pair2.
        if new_pair1 in self.weight_dict:
            gain += self.weight_dict.get_instance_pair_weight(new_pair1)
            gain += self.weight_dict.get_attribute_pair_weight(new_pair1)
            gain += self.weight_dict.get_circle_pair_weight(new_pair1)
            node_pair_dict = self.weight_dict.relation_pair_weight.get(new_pair1, {})
            for (x_node, y_node), weight in node_pair_dict.items():
                if new_mapping[x_node] == y_node:
                    self.log.debug("\trels_pair(%s-%s,%s-%s): %d"
                                   %(new_pair1[0], new_pair1[1], x_node, y_node,
                                     weight))
                    gain += weight
        if new_pair2 in self.weight_dict:
            gain += self.weight_dict.get_instance_pair_weight(new_pair2)
            gain += self.weight_dict.get_attribute_pair_weight(new_pair2)
            gain += self.weight_dict.get_circle_pair_weight(new_pair2)
            node_pair_dict = self.weight_dict.relation_pair_weight.get(new_pair2, {})
            for (x_node, y_node), weight in node_pair_dict.items():
                # Avoid duplicate.
                if x_node == new_pair1[0]:
                    continue
                elif new_mapping[x_node] == y_node:
                    self.log.debug("\trels_pair(%s-%s,%s-%s): %d"
                                   %(new_pair2[0], new_pair2[1], x_node, y_node,
                                     weight))
                    gain += weight

        # Deduct the triple match incurred by old_pair1 and old_pair2.
        if old_pair1 in self.weight_dict:
            gain -= self.weight_dict.get_instance_pair_weight(old_pair1)
            gain -= self.weight_dict.get_attribute_pair_weight(old_pair1)
            gain -= self.weight_dict.get_circle_pair_weight(old_pair1)
            node_pair_dict = self.weight_dict.relation_pair_weight.get(old_pair1, {})
            for (x_node, y_node), weight in node_pair_dict.items():
                if mapping[x_node] == y_node:
                    self.log.debug("\trels_pair(%s-%s,%s-%s): %d"
                                   %(old_pair1[0], old_pair1[1], x_node, y_node,
                                     weight))
                    gain -= weight
        if old_pair2 in self.weight_dict:
            gain -= self.weight_dict.get_instance_pair_weight(old_pair2)
            gain -= self.weight_dict.get_attribute_pair_weight(old_pair2)
            gain -= self.weight_dict.get_circle_pair_weight(old_pair2)
            node_pair_dict = self.weight_dict.relation_pair_weight.get(old_pair2, {})
            for (x_node, y_node), weight in node_pair_dict.items():
                # Avoid duplicate.
                if x_node == old_pair1[0]:
                    continue
                elif mapping[x_node] == y_node:
                    self.log.debug("\trels_pair(%s-%s,%s-%s): %d"
                                   %(old_pair2[0], old_pair2[1], x_node, y_node,
                                     weight))
                    gain -= weight
        self.match_triple_dict[new_mapping_tuple] = match_num + gain
        return gain

    def get_best_gain(self, mapping, cur_match_num):
        """Hill-climbing method to return the best gain swap/move can get.
        :param mapping: Current node mapping
        :param cur_match_num: current triple match number
        """
        largest_gain = 0
        # Record the nodes and action that bring the largest gain.
        node1, node2 = None, None
        use_swap = True

        # Compute move gain.
        unmatched_y_node_set = set(range(len(self.y_node2id)))
        for y_node in mapping:
            if y_node in unmatched_y_node_set:
                unmatched_y_node_set.remove(y_node)
        for x_node, y_node in enumerate(mapping):
            for new_y_node in unmatched_y_node_set:
                if new_y_node in self.candidate_mappings[x_node]:
                    #self.log.info("\tRemap node " + str(x_node) + " from " +
                    #              str(y_node) + " to " + str(new_y_node))
                    mv_gain = self.move_gain(
                        mapping, x_node, y_node, new_y_node, cur_match_num)
                    #self.log.info("\tMove gain:" + str(mv_gain))

                    #self.log.info("")

                    if mv_gain > largest_gain:
                        largest_gain = mv_gain
                        node1, node2 = x_node, new_y_node
                        use_swap = False

        # Compute swap gain.
        for x_node1, y_node1 in enumerate(mapping):
            # Constrain the order of node1 and node2 to avoid duplicates.
            for x_node2 in range(x_node1 + 1, len(mapping)):
                y_node2 = mapping[x_node2]

                #self.log.info("\tSwap node " + str(x_node1) + " and " + str(x_node2))
                #self.log.info("\tBefore swapping: %s <=> %s  %s <=> %s"
                #              % (x_node1, y_node1, x_node2, y_node2))

                sw_gain = self.swap_gain(
                    mapping, x_node1, y_node1, x_node2, y_node2, cur_match_num)
                #self.log.info("\tSwap gain: " + str(sw_gain))

                #self.log.info("")

                if sw_gain > largest_gain:
                    largest_gain = sw_gain
                    node1, node2 = x_node1, x_node2
                    use_swap = True

        # Generate a new mapping based on swap/move.
        cur_mapping = mapping[:]
        if node1 is not None:
            if use_swap:
                #self.log.info("Use swap gain")
                temp = cur_mapping[node1]
                cur_mapping[node1] = cur_mapping[node2]
                cur_mapping[node2] = temp
            else:
                #self.log.info("Use move gain")
                cur_mapping[node1] = node2
        else:
            pass
            #self.log.info("No move/swap gain found.")
        #self.log.info("")

        return largest_gain, cur_mapping

    def hill_climb(self, cur_mapping, match_num):
        #self.log.info("Start hill climbing.")
        while True:
            gain, new_mapping = self.get_best_gain(cur_mapping, match_num)
            #self.log.info("Gain after the hill-climbing:" + str(gain) + "\n")

            if gain <= 0:
                break

            match_num += gain
            cur_mapping = new_mapping[:]
            #self.log.info("Update triple match number to:" + str(match_num))
            #self.log.info("Current mapping:" + str(cur_mapping))
        return cur_mapping, match_num

    def print_mapping_weight(self, mapping):
        for key, value in enumerate(mapping):
            if value == -1:
                continue
            node_pair = (key, value)
            d1 = (self.weight_dict.get_instance_pair_weight(node_pair) +
                  self.weight_dict.get_attribute_pair_weight(node_pair) +
                  self.weight_dict.get_circle_pair_weight(node_pair))
            d2 = self.weight_dict.get_relation_pair_weight(node_pair, mapping)
            print ("\t%d-%d: %d %d" %(key, value, d1, d2))

    @classmethod
    def get_best_match(cls, instance1, attribute1, relation1,
                       instance2, attribute2, relation2, opts):
        """Get the highest triple match number between two sets of triples via
        hill-climbing.
        :param instance1: instance triples of representation1
                ("instance", node name, node value).
        :param attribute1: attribute triples of representation1
                (attribute name, node name, attribute value)
        :param relation1: relation triples of representation1
                (relation name, node 1 name, node 2 name)
        :param instance2: instance triples of representation2
                ("instance", node name, node value)
        :param attribute2: attribute triples of representation2
                (attribute name, node name, attribute value)
        :param relation2: relation triples of representation2
                (relation name, node 1 name, node 2 name)
        :param opts: Options.

        :return best_match: the node mapping that results in the highest triple matching number
        :return best_match_num: the highest triple matching number
        """
        utils.set_seed(opts.seed)
        s_metric = cls(
            instance1, attribute1, relation1,
            instance2, attribute2, relation2,
            iter_num=opts.iter_num,
            compute_instance=opts.compute_instance,
            compute_attribute=opts.compute_attribute,
            compute_relation=opts.compute_relation,
            mode=opts.mode
        )

        best_match_num = 0
        best_mapping = None
        for i in range(opts.iter_num):
            #s_metric.log.info("Iteration: %d" %i)
            if i == 0:
                # Do smart initialization at the first iteration.
                cur_mapping = s_metric.smart_init_mapping()
            elif opts.mode == TEST1:
                break
            else:
                # Do random initialization at the other iteration.
                cur_mapping = s_metric.random_init_mapping()
            # print (cur_mapping)
            # Compute the current triple match number.
            match_num = s_metric.compute_match(cur_mapping)
            #s_metric.log.info("Initial node mappings:\n" + str(cur_mapping))
            #s_metric.log.info("Initial triple matching number: %d\n" %(match_num))

            # Hill-climbing
            cur_mapping, match_num = s_metric.hill_climb(cur_mapping, match_num)

            # print (cur_mapping)
            if match_num > best_match_num:
                best_mapping = cur_mapping[:]
                best_match_num = match_num

        # s_metric.print_mapping_weight(cur_mapping)
        test_triple_num, gold_triple_num = 0, 0
        if opts.compute_instance:
            test_triple_num += len(instance1)
            gold_triple_num += len(instance2)
        if opts.compute_attribute:
            test_triple_num += len(attribute1)
            gold_triple_num += len(attribute2)
        if opts.compute_relation:
            test_triple_num += len(relation1)
            gold_triple_num += len(relation2)


        return best_mapping, best_match_num, test_triple_num, gold_triple_num


def normalize(item):
    """
    lowercase.
    """
    item = item.lower()
    return item

def compute_s_metric(true_graphs: List[DecompGraph],
                     pred_graphs: List[DecompGraph],
                     input_sents: List[str], 
                     semantics_only: bool,
                     drop_syntax: bool, 
                     include_attribute_scores: bool = False):
    """
    compute s-score between lists of decomp graphs
    """
    
    assert(len(true_graphs) == len(pred_graphs))

    # select DecompGraph or DecompGraphWithSyntax
    GraphType = None 
    if len(true_graphs) > 0:
        tg = true_graphs[0]
        if isinstance(tg, DecompGraph):
            GraphType = DecompGraph
        else:
            GraphType = DecompGraphWithSyntax

    else:
        return None

    total_match_num, total_test_num, total_gold_num = 0, 0, 0

    for g1, g2, sent  in tqdm(zip(pred_graphs, true_graphs, input_sents), total = len(true_graphs)):

        instances1, relations1, attributes1 = GraphType.get_triples(g1, 
                                                                    semantics_only, 
                                                                    drop_syntax, 
                                                                    include_attribute_scores = include_attribute_scores)

        instances1 = [Triple(x[1], x[0], x[2]) for x in instances1]
        attributes1 = [FloatTriple(x[0], x[1], x[2]) for x in attributes1]
        relations1 = [Triple(x[1], x[0], x[2]) for x in relations1]

        instances2, relations2, attributes2 = GraphType.get_triples(g2, 
                                                                    semantics_only, 
                                                                    drop_syntax, 
                                                                    include_attribute_scores = include_attribute_scores)


        instances2 = [Triple(x[1], x[0], x[2]) for x in instances2]
        attributes2 = [FloatTriple(x[0], x[1], x[2]) for x in attributes2]
        relations2 = [Triple(x[1], x[0], x[2]) for x in relations2]

        best_mapping, best_match_num, test_triple_num, gold_triple_num = S.get_best_match(
                instances1, attributes1, relations1,
                instances2, attributes2, relations2, c_args)

        #print('instances')
        #print([str(x) for x in instances1])
        #print([str(x) for x in instances2])

        #print('relations')
        #print([str(x) for x in relations1])
        #print([str(x) for x in relations2])

        #print(best_mapping)
        #print(f"match {best_match_num}")
        #print(f"test {test_triple_num}")
        #print(f"gold {gold_triple_num}")
        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num

    #print(f"total match {total_match_num} total_test {total_test_num} total gold {total_gold_num}") 
    precision, recall, best_f_score = utils.compute_f(
        total_match_num, total_test_num, total_gold_num)
    return precision, recall, best_f_score
