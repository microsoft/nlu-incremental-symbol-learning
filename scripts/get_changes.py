# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from collections import defaultdict
import argparse
import json
import pdb 


def main(args):
    incorrect_with_fxn_sets = {}
    incorrect_without_fxn_sets = {}
    correct_sets = {}
    dirs = ["5000_100", "10000_100", "20000_100", "50000_100", "100000_100", "max_100"]
    top_path = args.top_path
    dirs = [top_path + d for d in dirs]
    for dir in dirs:
        correct_set = [x.name for x in pathlib.Path(dir).joinpath("correct").glob("*")]
        incorrect_with_fxn_set = [x.name for x in pathlib.Path(dir).joinpath("incorrect_with_fxn").glob("*")]
        incorrect_without_fxn_set = [x.name for x in pathlib.Path(dir).joinpath("incorrect_without_fxn").glob("*")]

        correct_sets[dir] = set(correct_set)
        incorrect_with_fxn_sets[dir] = set(incorrect_with_fxn_set)
        incorrect_without_fxn_sets[dir] = set(incorrect_without_fxn_set)

    out_root = pathlib.Path(args.out_path)

    changed = []
    for i in range(1, len(dirs)):
        curr_cset = correct_sets[dirs[i]]
        prev_cset = correct_sets[dirs[i-1]]
        curr_iset_no_fxn = incorrect_without_fxn_sets[dirs[i]]
        curr_iset_with_fxn = incorrect_with_fxn_sets[dirs[i]]
        prev_iset_no_fxn = incorrect_without_fxn_sets[dirs[i-1]]
        prev_iset_with_fxn = incorrect_with_fxn_sets[dirs[i-1]]

        incorrect_no_fxn_to_correct = prev_iset_no_fxn & curr_cset
        correct_to_incorrect_no_fxn = prev_cset & curr_iset_no_fxn
        incorrect_with_fxn_to_correct = prev_iset_with_fxn & curr_cset
        correct_to_incorrect_with_fxn = prev_cset & curr_iset_with_fxn
        incorrect_no_fxn_to_incorrect_with_fxn = prev_iset_no_fxn & curr_iset_with_fxn
        incorrect_with_fxn_to_incorrect_no_fxn = prev_iset_with_fxn & curr_iset_with_fxn

        prev_dname = pathlib.Path(dirs[i-1]).name
        curr_dname = pathlib.Path(dirs[i]).name
        out_path = out_root.joinpath(f"{prev_dname}_to_{curr_dname}.json")
        to_write = dict(incorrect_no_fxn_to_correct = list(incorrect_no_fxn_to_correct),
                        correct_to_incorrect_no_fxn = list(correct_to_incorrect_no_fxn),
                        incorrect_with_fxn_to_correct = list(incorrect_with_fxn_to_correct),
                        correct_to_incorrect_with_fxn = list(correct_to_incorrect_with_fxn),
                        incorrect_no_fxn_to_incorrect_with_fxn = list(incorrect_no_fxn_to_incorrect_with_fxn),
                        incorrect_with_fxn_to_incorrect_no_fxn = list(incorrect_with_fxn_to_incorrect_no_fxn))

        with open(out_path, "w") as f1:
            json.dump(to_write, f1)

        #incorrect_to_correct = ", ".join(list(prev_iset & curr_cset))
        #correct_to_incorrect = ", ".join(list(prev_cset & curr_iset))

        #total_change = ", ".join(list(curr_cset ^ prev_cset))
        #for p in curr_cset ^ prev_cset:
        #    changed.append(p)

        #perc = len(curr_cset ^ prev_cset) / len(curr_cset | curr_iset)
        #cperc = len(prev_iset & curr_cset) / len(prev_iset)
        #iperc = len(prev_cset & curr_iset) / len(prev_cset)

        #print(f"from {dirs[i-1]} to {dirs[i]}")
        #print(f"the following changed: {total_change}, \n\twhich is {perc:.2f} of total programs")
        #print(f"became correct: {incorrect_to_correct}, \n\twhich is {cperc:.2f} of incorrect programs")
        #print(f"became incorrect: {correct_to_incorrect}, \n\twhich is {iperc:.2f} of correct programs")
        #print() 


    #print(f"total changes: {len(changed)}, unique changes {len(set(changed))}") 
    #change_count = defaultdict(int)
    #for c in changed:
    #    change_count[c]+=1
    #print(f"repeat offenders: {[(k,v) for k,v in change_count.items() if v > 1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-path", type=str, default="/home/t-eliass/scratch/error_analysis/Tomorrow")
    parser.add_argument("--out-path", type=str, default="/home/t-eliass/papers/incremental-function-parsing/results/error_analysis/Tomorrow/json")
    args = parser.parse_args()
    main(args)




