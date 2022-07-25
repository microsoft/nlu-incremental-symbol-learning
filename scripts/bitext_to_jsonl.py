# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import csv 
import argparse 
import json
import pathlib
from collections import defaultdict
import pdb 

def read_csv(path):
    to_ret = []
    with open(path) as f1:
        reader = csv.DictReader(f1)
        for line in reader:
            to_ret.append(line)
    return to_ret 

def write(data, path):
    with open(path, "w") as f1:
        for line in data:
            f1.write(json.dumps(line))

def read_txt(path):
    with open(path) as f1:
        data = f1.readlines()
    return data 

def main(args):
    src_text = read_txt(args.src_path)
    pred_text = read_txt(args.pred_path)
    metadata_lines = read_csv(args.metadata_path)

    assert(len(src_text) == len(pred_text) == len(metadata_lines))

    dialogues = defaultdict(lambda: defaultdict(dict))

    for src_line, pred_lispress, metadata in zip(src_text, pred_text, metadata_lines):
        dialogue_id = metadata['dialogue_id']
        turn_idx = metadata['turn_idx']

        dialogues[dialogue_id][int(turn_idx)] = {"src_line":src_line, 
                                            "pred_lispress": pred_lispress}

    

    with open(args.jsonl_path) as f1:
        data = [json.loads(x) for x in f1.readlines()]
    
    for i, jsonl in enumerate(data):
        for j, turn in enumerate(jsonl['turns']):
            corresponding = dialogues[jsonl['dialogue_id']][turn['turn_index']]
            # update with the prediction 
            if not turn['skip']:
                try:
                    turn['lispress'] = corresponding['pred_lispress']
                except KeyError:
                    pdb.set_trace()
            else:
                turn['lispress'] = "(ERROR)" 
            jsonl['turns'][j] = turn 
        data[i] = jsonl

    with open(args.out, "w") as f1:
        for line in data:
            f1.write(json.dumps(line) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=str, help="path .src file")
    parser.add_argument("--pred-path", type=str, help = "path to output .tgt file")
    parser.add_argument("--metadata-path", type=str, help = "path to metadata file")
    parser.add_argument("--jsonl-path", type=str, help = "path to input jsonl")
    parser.add_argument("--out", type=str, help = "path to output jsonl file")
    args = parser.parse_args() 
    main(args)
