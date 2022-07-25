# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb 
import pathlib
import re
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


FUNCTION = ['the','and','a','an','this','these']
np.random.seed(12)

def get_data(): 
    dataset = load_dataset("nlu_evaluation_data")
    train_dataset = dataset["train"]
    return train_dataset


def tokenize(data):
    def tokenize_helper(ex, i):
        ex['text_tokenized'] = re.split("[\s,.]+", ex['text'].lower().strip())
        ex['idx'] = i
        return ex
    data = [tokenize_helper(ex, i) for i, ex in enumerate(data)]
    return data

def get_probs(data, exclude=True, exclude_function=False):
    count_word = defaultdict(int)
    count_intent = defaultdict(int)
    count_intent_and_word = defaultdict(lambda: defaultdict(int))
    count_of_word_and_intent = defaultdict(lambda: defaultdict(int))
    for ex in data:
        intent = ex['label']
        count_intent[intent] += 1
        for tok in ex['text_tokenized']:
            count_word[tok] += 1
            count_intent_and_word[intent][tok] += 1
        for tok in set(ex['text_tokenized']):
            count_of_word_and_intent[intent][tok] +=1 

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


def get_max_probs(probs, k): 
    max_probs = {intent: None for intent in probs.keys()}
    for intent, prob_set in probs.items():
        top_k = sorted(list(prob_set.items()), key = lambda x: x[1])[-k:]
        max_probs[intent] = top_k
    return max_probs

def make_lookup_table(data, max_probs, threshold): 
    # {word: {intent: {example}}}
    lookup_table = defaultdict(lambda:  defaultdict(list))
    for intent in max_probs.keys():
        for trigger, prob in max_probs[intent]:
            if prob > threshold:
                for ex in data: 
                    if trigger in ex['text_tokenized']:
                        lookup_table[trigger][ex['label']].append(ex)
    return lookup_table


def mask_by_probability(tokenized_batch, tokenizer, probs_by_intent, probs_by_word, label_batch, text_batch, temperature=0.10, use_word = True, use_intent=False):
    MASK_ID=tokenizer.vocab[tokenizer._mask_token]
    MASK_TEXT=tokenizer._mask_token

    token_ids = tokenized_batch['input_ids']
    token_text = [tokenizer.convert_ids_to_tokens(x) for x in token_ids]

    for i, (tok_text, intent, text) in enumerate(zip(token_text, label_batch, text_batch)): 
        words = re.split("[\s,.]+", text)
        for word in words:
            if use_word:
                # get all probs that aren't that particular intent 
                all_other_probs = [prob for other_intent, prob in probs_by_word[word].items() if other_intent != intent]
                # get max prob; this makes it more likely that you'll mask a token if it's highly predictive of another intent 
                # e.g. if doing a non-radio example with "radio", it will be more likely to be masked 
                if len(all_other_probs) == 0:
                    prob_of_mask = 0
                else:
                    prob_of_mask = np.max(all_other_probs) * temperature
            elif use_intent:
                # get all probs that aren't that particular intent P(word | intent)
                all_other_probs = [probs_by_intent[other_intent][word] for other_intent in range(0, 68) if other_intent != intent]
                # get max prob p(word |intent); this makes it more likely that you'll mask a token if it's highly correlated
                # with another intent 
                # e.g. if doing a non-radio example with "radio", it will be more likely to be masked 
                if len(all_other_probs) == 0:
                    prob_of_mask = 0
                else:
                    prob_of_mask = np.max(all_other_probs) * temperature
            

            else:
                # or mask according to the inverse probability of the word with the intent
                # if the word is not very predictive of the intent, more likely to mask it 
                prob_intent_given_word = probs_by_word[word][intent]
                prob_of_mask = temperature * (1 - prob_intent_given_word)


            try:
                do_mask = np.random.choice([True, False], p=[prob_of_mask, 1-prob_of_mask])
            except ValueError:
                pdb.set_trace() 
            if do_mask:
                some_replaced = False
                # get subwords associated with word
                subword_ids = tokenizer(word)["input_ids"][1:-1]
                subwords = tokenizer.convert_ids_to_tokens(subword_ids)
                # check each subword span in the text 
                for span_start in range(0, len(tok_text)-len(subwords)):
                    span_end = span_start + len(subwords)
                    # if a span matches the subwords, replace it with a mask 
                    if tok_text[span_start:span_end] == subwords:
                        token_ids[i][span_start:span_end] = [MASK_ID for __ in range(len(subwords))]
                        token_text[i][span_start:span_end] = [MASK_TEXT for __ in range(len(subwords))]
                        some_replaced = True
        tokenized_batch['input_ids'] = token_ids
    return tokenized_batch  
                

if __name__ == "__main__": 
    #data = get_data()
    #data = tokenize(data)
    #probs = get_probs(data)
    #max_probs = get_max_probs(probs, k=3)
    #lookup_table = make_lookup_table(data, max_probs, 0.80)
    #pdb.set_trace() 

    batch_size = 3
    data = tokenize(get_data())
    probs = get_probs(data)
    probs_by_word = defaultdict(lambda: defaultdict(int))
    for intent in probs.keys():
        for word in probs[intent].keys():
            probs_by_word[word][intent] = probs[intent][word]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", add_special_tokens=True)
    curr_batch = {"input": [], "label": [], "input_str": []}
    curr_batch_as_text = {"input": [], "label": []}
    for chunk_start in range(0, len(data), batch_size):
        for example in data[chunk_start: chunk_start + batch_size]:
            label = example['label']
            text = example['text']
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        tokenized = mask_by_probability(tokenized, tokenizer, probs, curr_batch['label'], all_text) 
