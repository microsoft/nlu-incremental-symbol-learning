# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-

import sys
import re
import math



class BLEU:

    normalize1 = [
        ('<skipped>', ''),         # strip "skipped" tags
        (r'-\n', ''),              # strip end-of-line hyphenation and join lines
        (r'\n', ' '),              # join lines
        ('&quot;', '"'),           # avoid XML modules
    #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
    ]
    normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in normalize1]

    normalize2 = [
        (r'([\{-\~\[-\` -\&\(-\+\:-\@\/])',r' \1 '), # tokenize punctuation. apostrophe is missing
        (r'([^0-9])([\.,])',r'\1 \2 '),              # tokenize period and comma unless preceded by a digit
        (r'([\.,])([^0-9])',r' \1 \2'),              # tokenize period and comma unless followed by a digit
        (r'([0-9])(-)',r'\1 \2 ')                    # tokenize dash when preceded by a digit
    ]
    normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in normalize2]

    def __init__(self, n=4, preserve_case=False, nist_tokenize=True,
                 clip_len=False, eff_ref_len="shortest"):
        self.n = n
        self.preserve_case = preserve_case
        self.nist_tokenize = nist_tokenize
        self.clip_len = clip_len
        self.eff_ref_len = eff_ref_len

    def sentence_level_bleu(self, test, refs):
        refs = self.cook_refs(refs)
        test = self.cook_test(test, refs)
        return self.score_cooked([test])

    def count_ngrams(self, words):
        counts = {}
        for k in range(1,self.n+1):
            for i in range(len(words)-k+1):
                ngram = tuple(words[i:i+k])
                counts[ngram] = counts.get(ngram, 0)+1
        return counts

    def normalize(self, s):
        '''Normalize and tokenize text. This is lifted from NIST mteval-v11a.pl.'''
        if type(s) is not str:
            s = " ".join(s)
        # language-independent part:
        for (pattern, replace) in self.normalize1:
            s = re.sub(pattern, replace, s)
        # replaced with pattern in normalize1 to avoid pulling in XML
        # s = xml.sax.saxutils.unescape(s, {'&quot;':'"'})
        # language-dependent part (assuming Western languages):
        s = " %s " % s
        if not self.preserve_case:
            s = s.lower()         # this might not be identical to the original
        for (pattern, replace) in self.normalize2:
            s = re.sub(pattern, replace, s)
        return s.split()

    def cook_refs(self, refs):
        '''Takes a list of reference sentences for a single segment
        and returns an object that encapsulates everything that BLEU
        needs to know about them.'''

        if self.nist_tokenize:
            refs = [self.normalize(ref) for ref in refs]
        else:
            refs = [ref.split() for ref in refs]
            # refs = [[c for c in ref] for ref in refs]
        maxcounts = {}
        for ref in refs:
            counts = self.count_ngrams(ref)
            for (ngram,count) in counts.items():
                maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
        return ([len(ref) for ref in refs], maxcounts)

    def cook_test(self, test, refstats):
        '''Takes a test sentence and returns an object that
        encapsulates everything that BLEU needs to know about it.'''

        reflens, refmaxcounts = refstats
        if self.nist_tokenize:
            test = self.normalize(test)
        else:
            test = test.split()
            # test = [c for c in test]
        result = {}

        # Calculate effective reference sentence length.

        if self.eff_ref_len == "shortest":
            result["reflen"] = min(reflens)
        elif self.eff_ref_len == "average":
            result["reflen"] = float(sum(reflens))/len(reflens)
        elif self.eff_ref_len == "closest":
            result["reflen"] = min((abs(l-len(test)), l) for l in reflens)[1]
        else:
            sys.stderr.write("unknown effective reference length method: %s\n" % self.eff_ref_len)
            raise ValueError

        if self.clip_len:
            result["testlen"] = min(len(test), result["reflen"])
        else:
            result["testlen"] = len(test)

        result["guess"] = [max(0,len(test)-k+1) for k in range(1,self.n+1)]

        result['correct'] = [0]*self.n
        counts = self.count_ngrams(test)
        for (ngram, count) in counts.items():
            result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

        return result

    def score_cooked(self, allcomps):
        totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*self.n, 'correct':[0]*self.n}
        for comps in allcomps:
            for key in ['testlen','reflen']:
                totalcomps[key] += comps[key]
            for key in ['guess','correct']:
                for k in range(self.n):
                    totalcomps[key][k] += comps[key][k]
        logbleu = 0.0
        for k in range(self.n):
            if totalcomps['correct'][k] == 0:
                return 0.0
            logbleu += math.log(totalcomps['correct'][k])-math.log(totalcomps['guess'][k])
        logbleu /= float(self.n)
        logbleu += min(0,1-float(totalcomps['reflen'])/totalcomps['testlen'])
        return math.exp(logbleu)
