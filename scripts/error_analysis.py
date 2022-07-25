# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np 
import pdb 
import pathlib 
import os 
import re
import csv
import json 
from tqdm import tqdm 

from dataflow.core.lispress import parse_lispress, render_compact, render_pretty
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from miso.metrics.exact_match import AdvancedExactMatch

AMETRIC = AdvancedExactMatch()

def read_input(path):
    with open(path) as f1:
        return f1.readlines() 

def read_csv(path, delimiter = "\t"):
    to_ret = {}
    with open(path) as f1:
        for i, row in enumerate(csv.reader(f1, delimiter=delimiter)):
            if i == 0:
                continue
            to_ret[row[0]] = int(row[1])
    return to_ret

def read(path):
    with open(path) as f1:
        lines = [parse_lispress(x) for x in f1.readlines()]
    return lines

def read_synthetic(path):
    with open(path) as f1:
        return [x.strip() for x in f1.readlines()]

def read_tsv(path): 
    gold_data, pred_data, input_lines = [], [], []
    is_correct = []
    with open(path) as f1:
        reader = csv.reader(f1, delimiter=",") 
        for i, line in enumerate(reader):
            if i == 0:
                continue
            try:
                gold_str = render_compact(parse_lispress(line[2]))
            except:
                print(f"failed to parse gold {line[2]}")
                #pdb.set_trace()
            try:
                pred_str = render_compact(parse_lispress(line[3]))
            except:
                pred_str = render_compact(parse_lispress("( Error )"))

            gold_data.append(gold_str)
            pred_data.append(pred_str) 
            input_lines.append(line[1])
            is_correct.append(1 if line[-1] == "True" else 0)

    return gold_data, pred_data, input_lines 

def has_fxn(lispress, fxn = "PersonName.apply"):
    if type(lispress) != list:
        if lispress == fxn:
            return True
    else:
        return any([has_fxn(x, fxn) for x in lispress])

def get_all_person_names(lispress):
    names = []
    def get_pname_helper(lst):
        if type(lst) == list and lst[0] == "PersonName.apply": 
            try:
                name = re.sub('"', '', lst[1]).strip() 
            except:
                name = ""
            names.append(name)
        elif type(lst) != list:
            pass 
        else:
            [get_pname_helper(x) for x in lst]

    get_pname_helper(lispress)
    return names 

def check_all_fxn_acc(gold_data, pred_data, fxns, train_stats):
    by_fxn_correct = {fxn: 0 for fxn in fxns}
    train_totals = {}
    for fxn in train_stats.keys():
        train_totals[fxn] = train_stats[fxn]
    by_fxn_totals = {fxn: 0  for fxn in fxns}

    for fxn in tqdm(fxns):
        for i, (gd, pd) in enumerate(zip(gold_data, pred_data)):
            gs = render_compact(gd)
            ps = render_compact(pd)
            if has_fxn(gd, fxn = fxn):
                by_fxn_totals[fxn] += 1
                if AMETRIC(gs,ps):
                    by_fxn_correct[fxn] += 1

    to_ret = {}
    for fxn in by_fxn_correct.keys():
        try:
            train_total = by_fxn_totals[fxn]
        except KeyError:
            train_total = 0
        total = by_fxn_totals[fxn]
        if total == 0:
            total = 1
        to_ret[fxn] = (by_fxn_correct[fxn]/total, train_total)
    return to_ret

def check_synthetic_acc(gold_data, pred_data, fxn_of_interest="Func2"): 
    correct_examples, incorrect_examples_with_fxn, incorrect_examples_without_fxn = [], [], []
    for i, (gd, pd) in enumerate(zip(gold_data, pred_data)):
        whole_correct = gd == pd
        if fxn_of_interest not in gd.split(" "):
            continue
        if whole_correct: 
            correct_examples.append(i)
            continue
        else:
            if fxn_of_interest in pd.split(" "):
                incorrect_examples_with_fxn.append(i)
            else:
                incorrect_examples_without_fxn.append(i) 


    return 0, 0, correct_examples, incorrect_examples_with_fxn, incorrect_examples_without_fxn

def check_named_acc(gold_data, pred_data, fxn_of_interest=None):
    interest, non_interest = [], []
    debug_total = 0
    num_whole_correct = 0
    num_whole_correct_of_interest = 0
    total_of_interest = 0
    total = 0
    correct_examples, incorrect_examples_with_fxn, incorrect_examples_without_fxn = [], [], []
    for i, (gd, pd) in enumerate(zip(gold_data, pred_data)):
        total += 1
        whole_correct = False
        gs = render_compact(gd)
        ps = render_compact(pd)
        if AMETRIC(gs,ps):
            whole_correct = True
            num_whole_correct += 1

        if has_fxn(gd, fxn="PersonName.apply"):
            gold_names = get_all_person_names(gd)
            pred_names = get_all_person_names(pd)
            is_correct = 0
            if set(gold_names) == set(pred_names): 
                is_correct = 1
            if has_fxn(gd, fxn_of_interest):
                interest.append(is_correct)
            else:
                non_interest.append(is_correct)
        if has_fxn(gd, fxn = fxn_of_interest):
            if whole_correct:
                num_whole_correct_of_interest += 1
                correct_examples.append(i)
            else:
                if has_fxn(pd, fxn=fxn_of_interest):
                    incorrect_examples_with_fxn.append(i)
                else:
                    incorrect_examples_without_fxn.append(i)
            total_of_interest += 1

    print(f"total number {fxn_of_interest}: {total_of_interest}")
    print(f"number {fxn_of_interest} correct: {num_whole_correct_of_interest}") 
    print(f"percentage {fxn_of_interest} correct: {num_whole_correct_of_interest/total_of_interest*100:.2f}")   
    print() 
    print(f"number of interest {len(interest)}")
    print(f"number of rest {len(non_interest)}")

    fxn_named_acc = np.mean(interest)
    non_fxn_named_acc = np.mean(non_interest )
    print(f"{fxn_of_interest} accuracy on names: {fxn_named_acc*100:.2f}")
    print(f"Non-fxn accuracy on names: {non_fxn_named_acc*100:.2f}")

    return fxn_named_acc, non_fxn_named_acc, correct_examples, incorrect_examples_with_fxn, incorrect_examples_without_fxn

def write_examples(path, examples, pred_data, gold_data, input_lines):
    for i in examples:    
        p = path.joinpath(f"{i}")
        try:
            p.mkdir(parents=True, exist_ok=True) 
        except OSError:
            pass 

        with open(p.joinpath(f"gold.src"), "w") as inp_f, \
            open(p.joinpath("pred.pretty_tgt"),"w") as pred_f, \
            open(p.joinpath("gold.pretty_tgt"),"w") as gold_f,\
            open(p.joinpath("gold.tgt"),"w") as gold_f_compact:
            inp_f.write(input_lines[i])
            gold_f.write(render_pretty(gold_data[i]))
            pred_f.write(render_pretty(pred_data[i]))
            gold_f_compact.write(render_compact(gold_data[i]))

def main(args):
    if args.gold is not None:
        if not args.synthetic:
            gold_data = read(args.gold)
            pred_data = read(args.pred)
        else:
            gold_data = read_synthetic(args.gold)
            pred_data = read_synthetic(args.pred)
        input_lines = read_input(args.input)
    else:
        gold_data, pred_data, input_lines = read_tsv(args.from_tsv) 

    if args.fxn_of_interest is not None and not args.synthetic:
        fxn_named_acc, non_fxn_named_acc, correct_examples, incorrect_examples_with_fxn, incorrect_examples_without_fxn = check_named_acc(gold_data, pred_data, args.fxn_of_interest)
        print(f"{args.fxn_of_interest} accuracy on names: {fxn_named_acc*100:.2f}")
        print(f"Non-fxn accuracy on names: {non_fxn_named_acc*100:.2f}")
    elif args.fxn_of_interest is not None and args.synthetic:
        fxn_named_acc, non_fxn_named_acc, correct_examples, incorrect_examples_with_fxn, incorrect_examples_without_fxn = check_synthetic_acc(gold_data, pred_data, args.fxn_of_interest)
    elif args.all_fxns is not None:
        train_stats = read_csv(args.train_stats_file)
        all_fxns = open(args.all_fxns).read().split(",") 
        fxn_acc_dict = check_all_fxn_acc(gold_data, pred_data, all_fxns, train_stats) 
        with open(args.out_path, "w") as f1:
            json.dump(fxn_acc_dict, f1) 
    else:
        raise AssertionError("one of args.fxn_of_interest or args.all_fxns must be given") 

    correct_path = pathlib.Path(args.correct_output)
    incorrect_path_with_fxn = pathlib.Path(args.incorrect_output_with_fxn)
    incorrect_path_without_fxn = pathlib.Path(args.incorrect_output_without_fxn)
    write_examples(correct_path, correct_examples, pred_data, gold_data, input_lines)
    write_examples(incorrect_path_with_fxn, incorrect_examples_with_fxn, pred_data, gold_data, input_lines)
    write_examples(incorrect_path_without_fxn, incorrect_examples_without_fxn, pred_data, gold_data, input_lines)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gold", type=str, required=True, help="path to gold .tgt file") 
    parser.add_argument("--pred", type=str, required=True, help="path to pred .tgt file")
    parser.add_argument("--input",type=str, required=True, help="path to input src file")
    parser.add_argument("--fxn-of-interest", type=str, required=True, default=None)
    parser.add_argument("--all-fxns", type=str, required=False, default=None, help = "path to file containing comma-separated list of all functions") 
    parser.add_argument("--out-path", type=str, required=False, default=None, help="path to output json for all fxn accs") 
    parser.add_argument("--train-stats-file", type=str, required=False, default="../task_oriented_dialogue_as_dataflow_synthesis/output/onmt_data_stats/train.tgt.token_count.tsv")
    parser.add_argument("--correct-output", type=str, required=False, default="../scratch/correct")
    parser.add_argument("--incorrect-output-with-fxn", type=str, required=False, default="../scratch/incorrect_with_fxn")
    parser.add_argument("--incorrect-output-without-fxn", type=str, required=False, default="../scratch/incorrect_without_fxn")
    parser.add_argument("--synthetic", action="store_true") 
    args = parser.parse_args() 


    main(args)


