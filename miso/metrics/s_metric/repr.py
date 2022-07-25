# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from predpatt.patt import AMOD, APPOS, POSS
from predpatt.patt import Predicate
#from predpatt.repr.utils import get_var2insts, get_roots
#from predpatt.repr.utils import clean_edges, group_edges_by_pred, parse_edge
#from predpatt.repr import constants as constants

from miso.metrics.s_metric.bleu import BLEU

import numpy as np 

class Triple(object):

    def __init__(self, gov, rel, dep):
        self.gov = gov
        self.rel = rel
        self.dep = dep

    def position(self, x):
        return int(x.rsplit("/", 1)[1])

    def __str__(self):
        return "%s(%s, %s)" %(self.rel, self.gov, self.dep)

    def similarity(self, other):
        raise NotImplementedError

class FloatTriple(object):

    def __init__(self, attr_type, node, value):
        self.attr_type = attr_type
        self.node = node 
        self.value = value 

    def position(self, x):
        return int(x.rsplit("/", 1)[1])

    def __str__(self):
        return "%s(%s, %s)" %(self.attr_type, self.node, self.value)

    def similarity(self, other):
        # max diff = (-3-3)**2 since scores range [-3, 3]
        max_diff = 36
        if self.attr_type != other.attr_type:
            return 0
        # similarity 
        return (max_diff - (self.value - other.value)**2)/max_diff
        #return np.exp(-np.abs(self.value - other.value))

#class InstanceTriple(Triple):
#
#    bleu = BLEU(nist_tokenize=False)
#
#    def __init__(self, var, text, head_pos=-1, inst_pos=-1):
#        super(InstanceTriple, self).__init__(var, constants.INSTANCE, text)
#        self.head_pos = head_pos
#        self.inst_pos = inst_pos
#
#    def similarity(self, other):
#        x = self.dep.lower()
#        y = other.dep.lower()
#        n = max(len(x.split()), len(y.split()))
#        self.bleu.n = 4 if n > 4 else n
#        b = self.bleu.sentence_level_bleu(x, [y])
#        return b
#
#
#class InstanceTriple_TEST(InstanceTriple):
#
#    def __init__(self, var, text, head_pos=-1, inst_pos=-1):
#        super(InstanceTriple_TEST, self).__init__(var, text, head_pos, inst_pos)
#
#    def normalize(self, x):
#        """
#        Normalize node for sanity check between ref triples and triples recovered
#        from linearized representation.
#        """
#        if constants.HEAD in x:
#            return x.split(constants.HEAD)[0]
#        else:
#            return x.rsplit("/", 1)[0]
#
#    def similarity(self, other):
#        x = self.normalize(self.gov)
#        y = self.normalize(other.gov)
#        n = max(len(x.split()), len(y.split()))
#        self.bleu.n = 4 if n > 4 else n
#        b = self.bleu.sentence_level_bleu(x, [y])
#        return b
#
#
#class RelationTriple(Triple):
#
#    def __init__(self, gov, rel, dep):
#        super(RelationTriple, self).__init__(gov, rel, dep)
#
#    def __hash__(self):
#        return hash((self.gov, self.rel, self.dep))
#
#    def __eq__(self, other):
#        return (self.gov == other.gov and self.rel == other.rel and
#                self.dep == other.dep)
#
#    def __ne__(self, other):
#        return not (self == other)
#
#    def similarity(self, other):
#        return int(self.rel == other.rel)
#
#
#class RelationTriple_TEST(RelationTriple):
#
#    def __init__(self, gov, rel, dep):
#        super(RelationTriple_TEST, self).__init__(gov, rel, dep)
#
#    def similarity(self, other):
#        return int((self.rel.startswith("ARG") and other.rel.startswith("ARG"))
#                    or self.rel == other.rel)
#
#
#class NDTriples(object):
#
#    def __init__(self, instance_triples, relation_triples):
#        self.instance_triples = instance_triples
#        self.relation_triples = relation_triples
#
#    def __str__(self):
#        ret = ''
#        if len(self.instance_triples) == 0:
#            return ret
#        ret += "Instances:\n%s\n" %("\t" + "\n\t".join(map(str, self.instance_triples)))
#        if len(self.relation_triples) == 0:
#            return ret
#        ret += "Relations:\n%s\n" %("\t" + "\n\t".join(map(str, self.relation_triples)))
#        return ret
#
#    @staticmethod
#    def convert_from_predpatt(ppatt):
#        # Create the variable2instances dict.
#        var2insts = get_var2insts(ppatt)
#
#        # Create instance triples.
#        instance_triples = []
#        for var, instances in var2insts.items():
#            # Choose the instance which has the fewest tokens.
#            instance = min(instances, key=lambda x: len(x.phrase()))
#            instance_triples.append(InstanceTriple(var, instance.phrase(), instance.root.text))
#        instance_triples = sorted(instance_triples, key=lambda x: x.position(x.gov))
#
#        edges = clean_edges(ppatt.head_edges, var2insts)
#        roots, edges = get_roots(edges)
#        pred2edges = group_edges_by_pred(edges)
#        # Create relation triples from PredPatt instances.
#        relation_triples = []
#        for pred in pred2edges:
#            normal_argument_count = 0
#            for e in pred2edges[pred]:
#                predicate, relation, argument = parse_edge(e)
#                if relation in (constants.ARGUMENT, AMOD, APPOS, POSS, ppatt.ud.acl):
#                        relation_triples.append(
#                            RelationTriple(
#                                predicate.variable,
#                                constants.ARGUMENT + str(normal_argument_count),
#                                argument.variable)
#                        )
#                        normal_argument_count += 1
#                #elif relation == SUBPRED:
#                #    if not isinstance(argument, Predicate):
#                #        relation_triples.append(
#                #            RelationTriple(
#                #                predicate.variable,
#                #                constants.ARGUMENT + str(normal_argument_count),
#                #                argument.variable)
#                #        )
#                #        normal_argument_count += 1
#                else:
#                        relation_triples.append(
#                            RelationTriple(predicate.variable, relation, argument.variable)
#                        )
#
#        relation_triples = sorted(relation_triples, key=lambda x: (x.position(x.gov), x.position(x.dep)))
#        return NDTriples(instance_triples, relation_triples)
