# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb 
import pathlib
import re
import numpy as np 
from collections import defaultdict

FUNCTION = ['the','and','a','an','this','these']
np.random.seed(12)

def get_probs(instances, exclude=True, exclude_function=False):
    count_word = defaultdict(int)
    count_intent = defaultdict(int)
    count_intent_and_word = defaultdict(lambda: defaultdict(int))
    count_of_word_and_intent = defaultdict(lambda: defaultdict(int))
    for ex in instances:
        output_strs = ex['tgt_tokens_str'].metadata 
        input_strs  = ex['src_tokens_str'].metadata 

        for fxn in output_strs:

            count_intent[fxn] += 1
            for tok in input_strs:
                count_word[tok] += 1
                count_intent_and_word[fxn][tok] += 1
            for tok in set(ex['text_tokenized']):
                count_of_word_and_intent[fxn][tok] +=1 

    prob_intent_given_word = defaultdict(lambda: defaultdict(int))
    prob_word_given_intent = defaultdict(lambda: defaultdict(int))
    for intent in count_intent_and_word.keys():
        for word in count_intent_and_word[intent].keys():
            # Exclude anything that happens less than 10% of the time 
            if exclude and count_intent_and_word[intent][word] / count_intent[intent] < 0.10: 
                continue 
            if exclude_function and word.lower() in FUNCTION:
                continue

            single_prob_intent_given_word = count_intent_and_word[intent][word] / count_word[word]
            prob_intent_given_word[word][intent] = single_prob_intent_given_word
            single_prob_word_given_intent = count_of_word_and_intent[intent][word] / count_intent[intent]
            prob_word_given_intent[intent][word] = single_prob_word_given_intent

    return prob_intent_given_word, prob_word_given_intent