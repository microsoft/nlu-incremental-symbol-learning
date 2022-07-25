# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb 
import json
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm 

from data import split_by_intent
from source_lookup import tokenize, get_probs, get_max_probs, make_lookup_table, mask_by_probability, FUNCTION

def read_data(path):
    data = []
    with open(path + ".src_tok") as src_f, open(path + ".tgt") as tgt_f:
        src_data=src_f.readlines()
        tgt_data=tgt_f.readlines()
        for src, tgt in zip(src_data, tgt_data):
            example = {"text": src.strip(), "label": int(tgt.strip())}
            data.append(example)
    return data 

def main(args): 
    np.random.seed(args.seed)
    #if args.train_path is not None: 
    #    train_data = read_data(args.train_path)
    #if args.test_path is not None:
    #    test_data = read_data(args.test_path)

    dataset = load_dataset("nlu_evaluation_data")
    train_data, dev_data, test_data = split_by_intent(dataset, args.intent, args.n_data, args.n_intent)

    # get examples only with the trigger words but not the target intent 
    train_data = tokenize(train_data)
    test_data = tokenize(test_data)
    __, probs = get_probs(train_data, exclude_function=True)
    k = 3
    max_probs = get_max_probs(probs, k=k)
    print(max_probs.keys())
    intent = args.intent 
    triggers = max_probs[intent]
    print(triggers)
    examples_to_keep = []
    print(len(test_data))
    for example in tqdm(test_data):
        for trig in triggers: 
            if trig[0] in example['text_tokenized'] and example['label'] != intent:
                examples_to_keep.append(example)
    
    with open(args.out_path, "w") as f1:
        json.dump(examples_to_keep, f1)
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-data", type=int, default=750)
    parser.add_argument("--n-intent", type=int, default=15)
    parser.add_argument("--out-path", type=str, help="path to output json", required=True)
    parser.add_argument("--intent", type=int, default=50, help="intent of interest")
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    main(args)