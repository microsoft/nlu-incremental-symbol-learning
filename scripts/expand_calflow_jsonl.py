# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv 
import argparse 
import json


def read(path):
    to_ret = []
    with open(path) as f1:
        for line in f1:
            to_ret.append(json.loads(line))
    return to_ret 

def write(data, path):
    fieldnames = ['dialogue_id', 'turn_idx', 'has_exception', 'refer_are_correct']
    with open(path, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames) 
        writer.writeheader()
        for line in data:
            writer.writerow(line)

def main(args):
    data = read(args.path)
    to_write = []
    for i, line in enumerate(data):
        for j, turn in enumerate(line['turns']): 
            if turn['skip']:
                continue
            assert(j == turn['turn_index'])
            has_exception = int(turn['program_execution_oracle']['has_exception'])
            refer_are_correct = int(turn['program_execution_oracle']['refer_are_correct'])
            line_to_write = {"dialogue_id": line['dialogue_id'], 
                             "turn_idx": j, 
                             "has_exception": has_exception, 
                             "refer_are_correct": refer_are_correct} 
            to_write.append(line_to_write) 

    write(to_write, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to jsonlines file")
    parser.add_argument("--out", type=str, help = "path to output file")

    args = parser.parse_args() 
    main(args)
