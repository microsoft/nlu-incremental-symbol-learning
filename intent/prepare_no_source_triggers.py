# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import pathlib
import argparse 
import numpy as np 
from data import get_source_triggers, has_source_trigger
np.random.seed(12)

def prepare_source_triggers(data_path,
                            intent_of_interest,
                            source_triggers, 
                            out_path):

    data_path = pathlib.Path(data_path) 
    out_path = pathlib.Path(out_path) 
    out_path.mkdir(exist_ok=True, parents=True)

    with open(data_path.joinpath("train.json")) as train_f, \
         open(data_path.joinpath("dev.json")) as dev_f, \
         open(data_path.joinpath("test.json")) as test_f:
         train_data = json.load(train_f)
         dev_data = json.load(dev_f)
         test_data = json.load(test_f)

    if source_triggers is None:
        source_triggers = get_source_triggers(train_data, intent_of_interest)
    else:
        source_triggers = source_triggers.split(",")
    # filter not_interest so that source triggers don't appear 
    not_interest = [i for i in range(len(train_data)) if not has_source_trigger(train_data[i], source_triggers)]
    of_interest = [i for i, x in enumerate(train_data) if x['label'] == intent_of_interest]
    train_idxs = not_interest + of_interest 
    train_data = [train_data[idx] for idx in train_idxs]
    np.random.shuffle(train_data)

    with open(out_path.joinpath("train.json"), "w") as train_f, \
         open(out_path.joinpath("dev.json"), "w") as dev_f, \
         open(out_path.joinpath("test.json"), "w") as test_f:
         json.dump(train_data, train_f)
         json.dump(dev_data, dev_f)
         json.dump(test_data, test_f)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/nlu_eval_data")
    parser.add_argument("--intent-of-interest", default=None, type=int, help="intent to look at") 
    parser.add_argument("--source-triggers", type=str, default=None, help="source triggers to exclude in constructing the remainder of the dataset, e.g. radio,fm,am for play_radio intent. For analysis only.")
    args = parser.parse_args() 

    out_path = f"{args.data_path}_{args.intent_of_interest}_no_source" 

    prepare_source_triggers(args.data_path,
                            args.intent_of_interest,
                            args.source_triggers,
                            out_path) 
                            