# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python
# encoding: utf-8


class WeightDict(object):

    def __init__(self):
        self.instance_pair_weight = {}
        self.attribute_pair_weight = {}
        self.circle_pair_weight = {}
        self.relation_pair_weight = {}

    def __str__(self):
        return "\t" + "\n\t".join((
            "Instance pair weight: " + str(self.instance_pair_weight),
            "Attribute pair weight: " + str(self.attribute_pair_weight),
            "Circle pair weight: " + str(self.circle_pair_weight),
            "Relation pair weight: " + str(self.relation_pair_weight)))

    def __contains__(self, node_pair):
        return (node_pair in self.instance_pair_weight or node_pair in self.attribute_pair_weight or
                node_pair in self.relation_pair_weight or node_pair in self.circle_pair_weight)

    def add_node_pair(self, x_node, y_node, w, weight_dict):
        node_pair = (x_node, y_node)
        if node_pair not in weight_dict:
            weight_dict[node_pair] = 0
        weight_dict[node_pair] += w

    def add_instance_pair(self, x, y, w):
        self.add_node_pair(x, y, w, self.instance_pair_weight)

    def add_attribute_pair(self, x, y, w):
        self.add_node_pair(x, y, w, self.attribute_pair_weight)

    def add_circle_pair(self, x, y, w=1):
        self.add_node_pair(x, y, w, self.circle_pair_weight)

    def add_relation_pair(self, node_pair1, node_pair2, w=1):
        # if x.gov > x.dep:
        #     node_pair1 = (x.dep, y.dep)
        #     node_pair2 = (x.gov, y.gov)

        if node_pair1 in self.relation_pair_weight:
            if node_pair2 in self.relation_pair_weight[node_pair1]:
                self.relation_pair_weight[node_pair1][node_pair2] += w
            else:
                self.relation_pair_weight[node_pair1][node_pair2] = w
        else:
            self.relation_pair_weight[node_pair1] = {}
            self.relation_pair_weight[node_pair1][node_pair2] = w

        if node_pair2 in self.relation_pair_weight:
            if node_pair1 in self.relation_pair_weight[node_pair2]:
                self.relation_pair_weight[node_pair2][node_pair1] += w
            else:
                self.relation_pair_weight[node_pair2][node_pair1] = w
        else:
            self.relation_pair_weight[node_pair2] = {}
            self.relation_pair_weight[node_pair2][node_pair1] = w

    def node_pair_match_score(self, node_pair):
        if node_pair in self.instance_pair_weight:
            return self.instance_pair_weight[node_pair]
        return 0

    def get_instance_pair_weight(self, node_pair):
        if node_pair in self.instance_pair_weight:
            return self.instance_pair_weight[node_pair]
        return 0

    def get_attribute_pair_weight(self, node_pair):
        if node_pair in self.attribute_pair_weight:
            return self.attribute_pair_weight[node_pair]
        return 0

    def get_circle_pair_weight(self, node_pair):
        if node_pair in self.circle_pair_weight:
            return self.circle_pair_weight[node_pair]
        return 0

    def get_relation_pair_weight(self, node_pair1, mapping):
        if node_pair1 not in self.relation_pair_weight:
            return 0
        weight = 0
        for node_pair2 in self.relation_pair_weight[node_pair1]:
            if node_pair1[0] < node_pair2[0] and mapping[node_pair2[0]] == node_pair2[1]:
                # Only consider node_pair2 whose x_node is larger than node_pair1 to avoid duplicates
                # as we store both relation_pair_weight[node_pair1][node_pair2] and
                # relation_pair_weight[node_pair2][node_pair1].
                weight += self.relation_pair_weight[node_pair1][node_pair2]
        return weight
