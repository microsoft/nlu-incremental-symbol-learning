# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb 
from collections import defaultdict
import pathlib
import numpy as np
import torch 
import re
import json 
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset 
import sys 


from source_lookup import tokenize, get_probs, get_max_probs, make_lookup_table, mask_by_probability
np.random.seed(12) 

def random_split(dataset, p_train, p_dev, p_test): 
    dataset = dataset["train"]
    num_examples = len(dataset) 
    idxs = [i for i in range(num_examples)]
    np.random.shuffle(idxs)
    train_end = int(p_train * num_examples) 
    dev_start = train_end 
    dev_end = train_end + int( p_dev * num_examples) 
    test_start = dev_end 

    print(f"train end: {train_end}, dev end: {dev_end}") 
    train_idxs = idxs[0: train_end]
    dev_idxs = idxs[dev_start:dev_end]
    test_idxs = idxs[test_start:]
    train_data = [dataset[i] for i in train_idxs]
    dev_data = [dataset[i] for i in dev_idxs]
    test_data = [dataset[i] for i in test_idxs]
    return train_data, dev_data, test_data 

def has_source_trigger(datapoint, triggers):
    text = re.split("[\s,.]+", datapoint['text'].strip())
    for t in triggers:
        if "@@" in t: 
            # multi-gram 
            t_pieces = t.split("@@")
            for i in range(0, len(text) - len(t_pieces), len(t_pieces)): 
                ngram = text[i:i+len(t_pieces)]
                if all([t_pieces[j] == ngram[j] for j in range(len(ngram))]):
                    #print(" ".join(text ))
                    return True
        else:
            if t in text:
                return True
    return False

def filter_source_triggers(idxs, data, triggers): 
    to_ret = []
    for idx in idxs:
        if has_source_trigger(data[idx], triggers):
            continue
        else:
            to_ret.append(idx)
    if len(to_ret) == 0:
        raise AssertionError(f"There are 0 instances of the intent that do not have the triggers {', '.join(triggers)}")
    return to_ret 

def get_source_triggers(data, intent_of_interest): 
    data = tokenize(data)
    probs_intent_given_word, probs_words_given_intent = get_probs(data, exclude_function=True)

    probs_intent_given_word_by_intent = defaultdict(lambda: defaultdict(int))
    for word, intent_dict in probs_intent_given_word.items():
        for intent, prob in intent_dict.items():
            probs_intent_given_word_by_intent[intent][word] = prob
    top_3_forall = get_max_probs(probs_intent_given_word_by_intent, 3) 
    top_3_of_interest = top_3_forall[intent_of_interest]
    top_3_words = [x[0] for x in top_3_of_interest]
    return top_3_words


def get_adaptive_upsample_factor(of_interest, not_of_interest, dataset, intent_of_interest, eps=0.1):
    # get factor by which you need to upsample examples of intent_of_interest to maintain the roughly same source-target probabilities 
    train_idxs = of_interest + not_of_interest
    examples_of_interest = [dataset[i] for i in of_interest]
    examples_not_of_interest = [dataset[i] for i in not_of_interest]
    examples_of_interest = tokenize(examples_of_interest)
    examples_not_of_interest = tokenize(examples_not_of_interest)
    __, of_interest_probs_word_given_intent = get_probs(examples_of_interest, exclude=True, exclude_function=True)
    top_three_words = sorted(of_interest_probs_word_given_intent[intent_of_interest].items(), key=lambda x: x[1], reverse=True)[0:3]

    ratios = []
    for word, prob_of_interest in top_three_words:
        # count of word in of_interest 
        count_word_in_of_interest = np.sum([1 for example in examples_of_interest if word in example['text_tokenized']])
        count_word_in_not_of_interest = np.sum([1 for example in examples_not_of_interest if word in example['text_tokenized']])
        desired = prob_of_interest - eps
        minimum_count_of_interest = (count_word_in_not_of_interest * desired) / (1 - desired)
        ratios.append(minimum_count_of_interest/count_word_in_of_interest)
    return np.max(ratios)


def split_by_intent(data_path, 
                    intent_of_interest, 
                    n_data, 
                    n_intent, 
                    out_path = None, 
                    source_triggers = None,
                    do_source_triggers = False, 
                    upsample_by_factor=None, 
                    upsample_by_linear_fxn=False,
                    upsample_linear_fxn_coef=None,
                    upsample_linear_fxn_intercept=None,
                    upsample_constant_ratio=None,
                    upsample_constant_no_source=False,
                    adaptive_upsample=False, 
                    adaptive_factor=1.0): 
    #dataset = dataset['train']

    data_path = pathlib.Path(data_path) 
    with open(data_path.joinpath("train.json")) as train_f, \
         open(data_path.joinpath("dev.json")) as dev_f, \
         open(data_path.joinpath("test.json")) as test_f:
         train_data = json.load(train_f)
         dev_data = json.load(dev_f)
         test_data = json.load(test_f)

    # split into interest and non-interest 
    of_interest = [i for i, x in enumerate(train_data) if x['label'] == intent_of_interest]
    if n_intent > len(of_interest):
        # Take the max you can take while leaving some for test and dev
        num_devtest = int(0.3 * len(of_interest))
        num_train_intent = len(of_interest) - num_devtest
        n_intent = num_train_intent

    not_interest = [i for i in range(len(train_data)) if i not in of_interest]
    if n_data > n_intent + len(not_interest):
        # Take the max you can take 
        n_data = n_intent + len(not_interest)

    # NOTE: (elias) this is now deprecated because data with no source triggers is pre-determined and passed in via args.data_dir
    if do_source_triggers:
        if source_triggers is None:
            source_triggers = get_source_triggers(train_data, intent_of_interest)
        else:
            source_triggers = source_triggers.split(",")
        # filter not_interest so that source triggers don't appear 
        not_interest = [i for i in range(len(train_data)) if not has_source_trigger(train_data[i], source_triggers)]
    
    # NOTE: (elias) we don't actually want to shuffle, want to keep order the same 
    # np.random.shuffle(of_interest)
    # np.random.shuffle(not_interest)

    train_idxs = not_interest[0:n_data - n_intent] + of_interest[0:n_intent]
    if adaptive_upsample:
        upsample_by_factor = get_adaptive_upsample_factor(of_interest[0:n_intent], not_interest[0:n_data-n_intent], train_data, intent_of_interest) * adaptive_factor
        print(f"upsampling by a factor of {upsample_by_factor}")

    if upsample_by_linear_fxn:
        fxn_of_train = lambda x: upsample_linear_fxn_coef * x + upsample_linear_fxn_intercept
        upsample_by_factor = fxn_of_train(len(train_idxs))

    if upsample_by_factor is not None:
        effective_num_of_interest = int(len(of_interest[0:n_intent]) * upsample_by_factor)
        additional_num_of_interest =  effective_num_of_interest - len(of_interest[0:n_intent])
        sample_of_interest = np.random.choice(of_interest, size=additional_num_of_interest, replace=True).tolist()
        train_idxs += sample_of_interest

    if upsample_constant_ratio is not None: 
        effective_num_of_interest = int(len(train_idxs) * upsample_constant_ratio)
        additional_num_of_interest =  effective_num_of_interest - len(of_interest[0:n_intent])
        if upsample_constant_no_source:
            # need to remove things with source triggers from of_interest so that we don't also reduce source signal dilution
            of_interest = filter_source_triggers(of_interest, train_data, source_triggers)
        sample_of_interest = np.random.choice(of_interest, size=additional_num_of_interest, replace=True).tolist()
        train_idxs += sample_of_interest

    np.random.shuffle(train_idxs)


    train_data = [train_data[i] for i in train_idxs]
    #remaining = [i for i in range(len(dataset)) if i not in train_idxs]
    #np.random.shuffle(remaining)
    #dev_idxs = remaining[0:int(len(remaining)/2)]
    #test_idxs = remaining[int(len(remaining)/2):]
    #dev_data = [dataset[i] for i in dev_idxs]
    #test_data = [dataset[i] for i in test_idxs]

    n_interest_in_dev = np.sum([1 for x in dev_data if x['label'] == intent_of_interest])
    n_interest_in_test = np.sum([1 for x in test_data if x['label'] == intent_of_interest])
    print(f"There are {n_interest_in_dev} instances of {intent_of_interest} in dev and {n_interest_in_test} in test") 

    if out_path is not None:
        out_path = pathlib.Path(out_path)
        with open(out_path.joinpath("train.src_tok"), "w") as src_f, open(out_path.joinpath("train.tgt"), "w") as tgt_f:
            for i, datapoint in enumerate(train_data):
                src_f.write(datapoint['text'].strip() + "\n")
                tgt_f.write(str(datapoint['label']).strip() + "\n")

    return train_data, dev_data, test_data 

    

def batchify(data, batch_size, bert_model, device):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
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
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['input_str'] = all_text
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": [], "input_str": []}
        curr_batch_as_text = {"input": [], "label": []}
    return batches 


def batchify_min_pair(data, batch_size, bert_model, device, intent_of_interest, lookup_table):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}
    # add indices 
    data_lookup_dict = {}
    for i, datapoint in enumerate(data):
        datapoint['idx'] = i
        data[i] = datapoint
        data_lookup_dict[i] = (datapoint['text'].strip(), datapoint['label'])

    done = set()
    increased_batch_size = False
    for i, example in enumerate(data):
        # if current batch is right size, append and do next 
        if increased_batch_size or (len(curr_batch_as_text['input']) > 0 and len(curr_batch_as_text['input'])  % batch_size == 0):
            all_text = curr_batch_as_text['input'] 
            tokenized = tokenizer(all_text,  padding=True)
            ids = tokenized['input_ids']
            curr_batch['input'] = ids
            curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
            curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

            batches.append(curr_batch)
            curr_batch = {"input": [], "label": []}
            curr_batch_as_text = {"input": [], "label": []}
            # if batch size was increased in earlier iteration, decrease again 
            if increased_batch_size:
                batch_size -= 1
                increased_batch_size = False 

        # don't use same example twice
        if example['idx'] in done:
            continue

        # get an example 
        label = example['label']
        text = example['text']
        # if it's an example of interest, get its min pair
        if label == intent_of_interest:
            top_pair_idxs = lookup_table[i]
            i = 0
            while  top_pair_idxs[i] in done:
                i+=1
            tp_idx = top_pair_idxs[i]

        # edge case: the example of interest is last in the batch, 
        # and adding one more will put over batch size, then increase 
        # batch size by 1 temporarily 
        if (len(curr_batch_as_text['input']) + 1) % batch_size == 0:
            batch_size += 1
            increased_batch_size = True

        # Add the example 
        curr_batch_as_text['input'].append(text)
        curr_batch['label'].append(label)
        done.add(example['idx'])
        # if appropriate, add min pair 
        if label == intent_of_interest:
            top_pair = data_lookup_dict[tp_idx]
            print(f"batching {text}:{label} with {top_pair[0]}:{top_pair[1]}")
            curr_batch_as_text['input'].append(top_pair[0])
            curr_batch['label'].append(top_pair[1])
            done.add(tp_idx)

    # add last batch
    if len(curr_batch_as_text['input']) > 0: 
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        
    return batches 

def batchify_double_in_batch(data, batch_size, bert_model, device, intent_of_interest):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}

    increased_batch_size = False
    for i, example in enumerate(data):
        # if current batch is right size, append and do next 
        if increased_batch_size or (len(curr_batch_as_text['input']) > 0 and len(curr_batch_as_text['input'])  % batch_size == 0):
            all_text = curr_batch_as_text['input'] 
            tokenized = tokenizer(all_text,  padding=True)
            ids = tokenized['input_ids']
            curr_batch['input'] = ids
            curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
            curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

            batches.append(curr_batch)
            curr_batch = {"input": [], "label": []}
            curr_batch_as_text = {"input": [], "label": []}
            # if batch size was increased in earlier iteration, decrease again 
            if increased_batch_size:
                batch_size -= 1
                increased_batch_size = False 

        # get an example 
        label = example['label']
        text = example['text']
        curr_batch_as_text['input'].append(text)
        curr_batch['label'].append(label)

        # if it's an example of interest, double it 
        if label == intent_of_interest:
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)

        # edge case: the example of interest is last in the batch, 
        # and adding one more will put over batch size, then increase 
        # batch size by 1 temporarily 
        if (len(curr_batch_as_text['input']) + 1) % batch_size == 0:
            batch_size += 1
            increased_batch_size = True

    # add last batch
    if len(curr_batch_as_text['input']) > 0: 
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        
    return batches 


def batchify_double_in_data(data, batch_size, bert_model, device, intent_of_interest):
    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}

    # double in data
    new_data =[]
    for i, example in enumerate(data):
        new_data.append(example)
        if example['label'] == intent_of_interest:
            new_data.append(example)

    # shuffle 
    np.random.shuffle(new_data)

    for chunk_start in range(0, len(new_data), batch_size):
        for example in data[chunk_start: chunk_start + batch_size]:
            label = example['label']
            text = example['text']
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": []}
        curr_batch_as_text = {"input": [], "label": []}
    return batches 


def batchify_by_source_trigger(data, batch_size, bert_model, device, intent_of_interest, k=3, threshold=0.80): 


    data = tokenize(data)
    probs = get_probs(data)
    max_probs = get_max_probs(probs, k=k)
    lookup_table = make_lookup_table(data, max_probs, threshold=threshold)

    batches = []
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": []}
    curr_batch_as_text = {"input": [], "label": []}
    # shuffle 
    np.random.shuffle(data)
    done = []
    evicted = []
    for chunk_start in range(0, len(data), batch_size):
        for i, example in enumerate(data[chunk_start: chunk_start + batch_size]):
            if example['idx'] in done:
                continue 

            label = example['label']
            text = example['text']
            # if we have an intent, only perform for the intent of interest 
            if intent_of_interest is not None and label == intent_of_interest:
                # perform lookup for just intent of interest 
                max_words = [w for w, p in max_probs[label] if p > threshold]
                do_break = False
                for word in max_words:
                    other_intents = set(lookup_table[word].keys()) - set([label])
                    for o_int in other_intents:
                        examples = lookup_table[word][o_int]
                        for ex in examples:
                            if ex['idx'] not in done:
                                # found a new one, add to the batch 
                                print(f"batching together the following based on {word}:")
                                print(f"\t{example['text']}: {example['label']}")
                                print(f"\t{ex['text']}: {ex['label']}")
                                curr_batch_as_text['input'].append(ex['text'])
                                curr_batch['label'].append(ex['label'])
                                done.append(ex['idx'])
                                do_break = True
                                break
                        if do_break:
                            break
                    if do_break:
                        break


            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)
            done.append(example['idx'])

            if len(curr_batch_as_text['input']) == batch_size:
                # we're full, remaining are evicted
                num_missing = batch_size - i
                curr_end = chunk_start + batch_size - num_missing
                curr_evicted = data[curr_end:chunk_start + batch_size]
                evicted += curr_evicted 
                break


        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)
        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": []}
        curr_batch_as_text = {"input": [], "label": []}


    # iterate over evicted examples and add them back in 
    if len(evicted) > 0:
        for chunk_start in range(0, len(evicted), batch_size):
            curr_batch = {"input": [], "label": []}
            curr_batch_as_text = {"input": [], "label": []}
            for i, example in enumerate(evicted[chunk_start: chunk_start + batch_size]):
                try:
                    assert(example['idx'] not in done)
                except AssertionError:
                    continue
                curr_batch_as_text['input'].append(ex['text'])
                curr_batch['label'].append(ex['label'])

            all_text = curr_batch_as_text['input'] 
            if len(all_text) == 0:
                continue
            try:
                tokenized = tokenizer(all_text,  padding=True)
            except AssertionError:
                pdb.set_trace() 
            ids = tokenized['input_ids']
            curr_batch['input'] = ids
            curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
            curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)
            batches.append(curr_batch)

    return batches  

def batchify_mask_source_trigger(data, batch_size, bert_model, device, temperature, use_word, use_intent): 
    """
    batchify and mask tokens according to the inverse of their prob of appearing with a given token
    idea is that "radio" should be more likely to be masked with non-radio intent, very unlikely with radio intent  
    """

    batches = []
    data = tokenize(data)
    probs_by_word, probs_by_intent  = get_probs(data, exclude=False)
    # probs_by_word, probs_by_intent = defaultdict(lambda: defaultdict(int))
    # for intent in probs_by_intent.keys():
    #     for word in probs_by_intent[intent].keys():
    #         probs_by_word[word][intent] = probs_by_intent[intent][word]

    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
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
        tokenized = mask_by_probability(tokenized, tokenizer, probs_by_intent, probs_by_word,  curr_batch['label'], all_text, temperature, use_word, use_intent) 

        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['input_str'] = all_text
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": [], "input_str": []}
        curr_batch_as_text = {"input": [], "label": []}
    return batches 

def batchify_weight_source_trigger(data, batch_size, bert_model, device, intent_of_interest, temperature): 
    """
    batchify and mask tokens according to the inverse of their prob of appearing with a given token
    idea is that "radio" should be more likely to be masked with non-radio intent, very unlikely with radio intent  
    """

    batches = []
    data = tokenize(data)
    probs_by_word, probs_by_intent  = get_probs(data, exclude=False)

    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": [], "input_str": [], "weight": []}
    curr_batch_as_text = {"input": [], "label": []}
    for chunk_start in range(0, len(data), batch_size):
        for example in data[chunk_start: chunk_start + batch_size]:
            label = example['label']
            text = example['text']
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)
            probs_word_given_intent = [probs_by_intent[intent_of_interest][tok] for tok in example['text_tokenized']]
            if len(probs_word_given_intent) == 0:
                weight = 1.0
            else:
                weight = 1 - np.max(probs_word_given_intent) * temperature
            curr_batch['weight'].append(weight)

        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)

        #tokenized = mask_by_probability(tokenized, tokenizer, probs_by_intent, probs_by_word,  curr_batch['label'], all_text, temperature, use_word, use_intent) 

        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['input_str'] = all_text
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)
        curr_batch['weight'] = torch.tensor(curr_batch['weight']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": [], "input_str": [], "weight": []}
        curr_batch_as_text = {"input": [], "label": []}
    return batches 

def batchify_sample_source_trigger(data, batch_size, bert_model, device, intent_of_interest=None, temperature=1.0): 
    """
    batchify by sampling examples according to the inverse of their prob of appearing with a given token
    idea is that "radio" should be more likely to be sampled with radio intent early in training, and then later in
    training start adding in competing intents 
    """

    batches = []
    data = tokenize(data)
    probs_intent_given_word, probs_word_given_intent = get_probs(data, exclude=False)
    INTENTS = set([i for i in range(69)])
    tokenizer = AutoTokenizer.from_pretrained(bert_model, add_special_tokens=True)
    curr_batch = {"input": [], "label": [], "input_str": []}
    curr_batch_as_text = {"input": [], "label": []}
    threshold = 0.50

    for example in data:
        label = example['label']
        remaining_intents  = INTENTS - set([label])
        # get the max probability P(word | intent) across all other intents and all words in the sentence 
        # so for example if the word "radio" is in the example and the example is not intent 50, max prob will be high
        try:
            max_prob_word_given_intent = np.max([probs_word_given_intent[intent][tok]  for intent in remaining_intents for tok in example['text_tokenized'] if probs_intent_given_word[tok][intent] > threshold])
        except ValueError:
            max_prob_word_given_intent = 0.0 
        # this will be the inverse probability times the temperature
        # when the temp is high and the max prob is high, unlikely to be included
        # as training goes on, temp decreases, more likely to be included 
        prob_include_in_epoch = 1-max_prob_word_given_intent * temperature
        include_in_epoch = np.random.choice([True, False], p=[prob_include_in_epoch, 1-prob_include_in_epoch])

        # only include 
        if include_in_epoch:
            text = example['text']
            curr_batch_as_text['input'].append(text)
            curr_batch['label'].append(label)

        if len(curr_batch_as_text['input']) == batch_size:
            all_text = curr_batch_as_text['input'] 
            tokenized = tokenizer(all_text,  padding=True)

            ids = tokenized['input_ids']
            curr_batch['input'] = ids
            curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
            curr_batch['input_str'] = all_text
            curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

            batches.append(curr_batch)
            curr_batch = {"input": [], "label": [], "input_str": []}
            curr_batch_as_text = {"input": [], "label": []}

    # get leftovers 
    if len(curr_batch_as_text['input']) > 0:
        all_text = curr_batch_as_text['input'] 
        tokenized = tokenizer(all_text,  padding=True)

        ids = tokenized['input_ids']
        curr_batch['input'] = ids
        curr_batch['input'] = torch.tensor(curr_batch['input']).to(device)
        curr_batch['input_str'] = all_text
        curr_batch['label'] = torch.tensor(curr_batch['label']).to(device)

        batches.append(curr_batch)
        curr_batch = {"input": [], "label": [], "input_str": []}
        curr_batch_as_text = {"input": [], "label": []}

    return batches 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path")
    args= parser.parse_args()

    dataset = load_dataset("nlu_evaluation_data")

    train_data, dev_data, test_data = random_split(dataset, 0.7, 0.1, 0.2)

    out_path = pathlib.Path(args.out_path)
    
    with open(out_path.joinpath("train.json"), "w") as train_f, open(out_path.joinpath("dev.json"), "w") as dev_f, open(out_path.joinpath("test.json"), "w") as test_f: 
        json.dump(train_data, train_f)
        json.dump(dev_data, dev_f)
        json.dump(test_data, test_f)
