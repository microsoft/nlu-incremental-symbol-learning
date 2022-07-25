# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import argparse
from dataflow.core.lispress import parse_lispress, program_to_lispress, lispress_to_program
from dataflow.core.program import ValueOp, Expression
from dataflow.core.linearize import lispress_to_seq, seq_to_lispress
import numpy as np
from tqdm import tqdm 
import pdb 
import time
import re 
import string
import json
from copy import deepcopy
import Levenshtein as lev


def anonymize(lispress_seq):
    lispress = seq_to_lispress(lispress_seq)
    program, __ = lispress_to_program(lispress, 0)

    for i, expr in enumerate(program.expressions):
        op = expr.op
        if isinstance(op, ValueOp):
            value = json.loads(op.value)
            if value['schema'] == "String":
                value['underlying'] = "String"
            elif value['schema'] == "Long": 
                value['underlying'] = 0
            elif value['schema'] == "Boolean":
                value['underlying'] = True
            elif value['schema'] == "Number":
                value['underlying'] = 0
            else:
                pdb.set_trace()
            new_op = ValueOp(json.dumps(value))
            new_expr = Expression(expr.id, new_op, expr.type_args, expr.type, expr.arg_ids)
            program.expressions[i] = new_expr 
    
    lispress = program_to_lispress(program)
    seq = lispress_to_seq(lispress)
    return seq


def read_and_split(gold_path, pred_path, fxn_of_interest):
    with open(gold_path) as f1, open(pred_path) as f2:
        gold_lines = [x for x in f1.readlines()]
        pred_lines = [x for x in f2.readlines()]

    gold_src_path = re.sub("\.tgt", ".src_tok", str(gold_path))

    with open(gold_src_path) as f1:
        gold_src_lines = [x for x in f1.readlines()]

    (correct_gold_fxn_lines, 
     incorrect_gold_fxn_lines, 
     correct_pred_fxn_lines, 
     incorrect_pred_fxn_lines) = [], [], [], []

    (correct_gold_src_lines, 
     incorrect_gold_src_lines) = [], []

    for i, (gline, gsrc, pline) in enumerate(zip(gold_lines, gold_src_lines, pred_lines)):
        gline_lsp = parse_lispress(gline)
        gline = lispress_to_seq(gline_lsp)
        try:
            pline_lsp = parse_lispress(pline)
            pline = lispress_to_seq(pline_lsp)
        except: 
            print(f"Error on {pline}")
            pline = "( Error )"
        gline_split = gline
        pline_split = pline
        if fxn_of_interest is not None:
            if fxn_of_interest in gline_split:
                if pline == gline:
                    correct_gold_src_lines.append(gsrc)
                    correct_gold_fxn_lines.append(gline_split)
                    correct_pred_fxn_lines.append(pline_split) 
                else:
                    incorrect_gold_src_lines.append(gsrc)
                    incorrect_gold_fxn_lines.append(gline_split)
                    incorrect_pred_fxn_lines.append(pline_split)
        else:
            if pline == gline:
                correct_gold_src_lines.append(gsrc)
                correct_gold_fxn_lines.append(gline_split)
                correct_pred_fxn_lines.append(pline_split) 
            else:
                incorrect_gold_src_lines.append(gsrc)
                incorrect_gold_fxn_lines.append(gline_split)
                incorrect_pred_fxn_lines.append(pline_split)

    return correct_gold_fxn_lines, incorrect_gold_fxn_lines, correct_pred_fxn_lines, incorrect_pred_fxn_lines, correct_gold_src_lines, incorrect_gold_src_lines

def read_train(path, fxn_of_interest): 
    with open(path) as f1:
        lines = [x for x in f1.readlines()]

    src_path = re.sub("\.tgt", ".src_tok", str(path))
    with open(src_path) as f1:
        src_lines = [x.strip() for x in f1.readlines()]

    to_ret_tgt, to_ret_src = [], []
    for line, src_line in zip(lines, src_lines):
        split_line = line.strip().split(" ")
        if fxn_of_interest is not None:
            if fxn_of_interest in split_line:
                to_ret_tgt.append(split_line)
                to_ret_src.append(src_line)
        else:
            to_ret_tgt.append(split_line)
            to_ret_src.append(src_line)
    return to_ret_tgt, to_ret_src

def get_levenshteins(list_a, list_b, do_anonymize = False):
    print(f"Getting Levenshtein for {len(list_a)} X {len(list_b)} = {len(list_a) * len(list_b)} examples")
    to_ret = np.ones((len(list_a), len(list_b))) * np.inf
    for i, seq_a in tqdm(enumerate(list_a)):
        if do_anonymize:
            seq_a = anonymize(seq_a)
        for j, seq_b in enumerate(list_b):
            if do_anonymize:
                seq_b = anonymize(seq_b)
            ls = levenshtein(seq_a, seq_b)
            to_ret[i,j] = ls
    return to_ret 

def levenshtein(s1, s2): 
    """Compute Levenshtein distance"""
    # need to turn s1 and s2 vocabs into characters 
    total_vocab = set(s1) | set(s2)
    chars = string.ascii_letters + string.digits + string.punctuation
    try:
        assert(len(total_vocab) < len(chars))
    except AssertionError:
        print("Warning: mapping incomplete, returning large distance to ignore in min")
        return np.inf
    chars = chars[0:len(total_vocab)]
    total_vocab = list(total_vocab)
    mapping = {k:c for k, c in zip(total_vocab, chars)}

    s1 = [mapping[x] for x in s1]
    s2 = [mapping[x] for x in s2]
    s1 = "".join(s1)
    s2 = "".join(s2)

    return lev.distance(s1, s2)

def run_main(gold_path, pred_path, train_path, fxn_of_interest, do_anonymize=False):
    if type(fxn_of_interest) == tuple:
        first_fxn, second_fxn = fxn_of_interest

    else:
        first_fxn = fxn_of_interest
        second_fxn = fxn_of_interest

    correct_golds, incorrect_golds, correct_preds, incorrect_preds, correct_gold_src, incorrect_gold_src = read_and_split(gold_path, pred_path, first_fxn)
    all_train, __ = read_train(train_path, second_fxn)

    # get levenshtein distance of each example to each FindManager example in the train set 
    correct_levenshteins = get_levenshteins(correct_golds, all_train, do_anonymize)
    incorrect_levenshteins = get_levenshteins(incorrect_golds, all_train, do_anonymize)
    # take min 
    min_correct = np.min(correct_levenshteins, axis=1)
    min_incorrect = np.min(incorrect_levenshteins, axis=1)
    mean_correct = np.mean(min_correct)
    mean_incorrect = np.mean(min_incorrect)

    return mean_correct, mean_incorrect


def run_main_two_groups(gold_path, pred_path, train_path, out_path, fxn_of_interest, do_anonymize=True):
    if type(fxn_of_interest) == tuple:
        first_fxn, second_fxn = fxn_of_interest

    else:
        first_fxn = fxn_of_interest
        second_fxn = fxn_of_interest

    correct_golds, incorrect_golds, correct_preds, incorrect_preds, correct_gold_src, incorrect_gold_src = read_and_split(gold_path, pred_path, first_fxn)
    all_train, all_train_src = read_train(train_path, second_fxn)

    # get levenshtein distance of each example to each FindManager example in the train set 
    correct_levenshteins = get_levenshteins(correct_golds, all_train, do_anonymize=True)
    incorrect_levenshteins = get_levenshteins(incorrect_golds, all_train, do_anonymize=True)


    # take min 
    min_correct_idxs = np.argmin(correct_levenshteins, axis = 1)
    min_incorrect_idxs = np.argmin(incorrect_levenshteins, axis = 1)

    min_correct = np.take_along_axis(correct_levenshteins, min_correct_idxs.reshape(-1,1), axis=1)
    min_incorrect = np.take_along_axis(incorrect_levenshteins, min_incorrect_idxs.reshape(-1,1), axis=1) 

    # split into ones with 0 distance and others 
    correct_idxs_with_zero = [i for i in range(len(min_correct)) if min_correct[i] == 0]
    incorrect_idxs_with_zero = [i for i in range(len(min_incorrect)) if min_incorrect[i] == 0]

    correct_train_idxs_with_zero = min_correct_idxs[correct_idxs_with_zero]
    incorrect_train_idxs_with_zero = min_incorrect_idxs[incorrect_idxs_with_zero]

    correct_train_with_zero = {i:all_train[i] for i in correct_train_idxs_with_zero}
    correct_train_src_with_zero = {i:all_train_src[i] for i in correct_train_idxs_with_zero}
    incorrect_train_with_zero = {i:all_train[i] for i in incorrect_train_idxs_with_zero}
    incorrect_train_src_with_zero = {i:all_train_src[i] for i in incorrect_train_idxs_with_zero}


    correct_idxs_with_nonzero = [i for i in range(len(min_correct)) if i not in correct_idxs_with_zero]
    incorrect_idxs_with_nonzero = [i for i in range(len(min_incorrect)) if i not in incorrect_idxs_with_zero]

    correct_train_idxs_with_nonzero = min_correct_idxs[correct_idxs_with_nonzero]
    incorrect_train_idxs_with_nonzero = min_incorrect_idxs[incorrect_idxs_with_nonzero]

    correct_train_with_nonzero = {i:all_train[i] for i in correct_train_idxs_with_nonzero}
    correct_train_src_with_nonzero = {i:all_train_src[i] for i in correct_train_idxs_with_nonzero}
    incorrect_train_with_nonzero = {i:all_train[i] for i in incorrect_train_idxs_with_nonzero}
    incorrect_train_src_with_nonzero = {i:all_train_src[i] for i in incorrect_train_idxs_with_nonzero}

    with open(out_path.joinpath("correct_zero_gold.tgt"), "w") as gold_f, \
         open(out_path.joinpath("correct_zero_pred.tgt"), "w") as pred_f, \
         open(out_path.joinpath("correct_zero_gold_train.src_tok"), "w") as gold_src_train_f, \
         open(out_path.joinpath("correct_zero_gold_test.src_tok"), "w") as gold_src_test_f:
        for correct_idx in correct_idxs_with_zero:
            gold_plan = correct_train_with_zero[min_correct_idxs[correct_idx]]
            gold_src = correct_train_src_with_zero[min_correct_idxs[correct_idx]]
            pred_plan = correct_preds[correct_idx]
            test_src = correct_gold_src[correct_idx]
            gold_src_train_f.write(f"{gold_src.strip()}\n")
            gold_src_test_f.write(f"{test_src.strip()}\n")
            gold_f.write(f"{' '.join(gold_plan)}\n")
            pred_f.write(f"{' '.join(pred_plan)}\n")

    with open(out_path.joinpath("incorrect_zero_gold.tgt"), "w") as gold_f, \
        open(out_path.joinpath("incorrect_zero_pred.tgt"), "w") as pred_f, \
        open(out_path.joinpath("incorrect_zero_gold_train.src_tok"), "w") as gold_src_train_f, \
        open(out_path.joinpath("incorrect_zero_gold_test.src_tok"), "w") as gold_src_test_f:
        for incorrect_idx in incorrect_idxs_with_zero:
            gold_plan = incorrect_train_with_zero[min_incorrect_idxs[incorrect_idx]]
            pred_plan = incorrect_preds[incorrect_idx]
            gold_src = incorrect_train_src_with_zero[min_incorrect_idxs[incorrect_idx]]
            test_src = incorrect_gold_src[incorrect_idx]
            gold_src_train_f.write(f"{gold_src.strip()}\n")
            gold_src_test_f.write(f"{test_src.strip()}\n")
            gold_f.write(f"{' '.join(gold_plan)}\n")
            pred_f.write(f"{' '.join(pred_plan)}\n")

    with open(out_path.joinpath("correct_nonzero_gold.tgt"), "w") as gold_f, \
         open(out_path.joinpath("correct_nonzero_pred.tgt"), "w") as pred_f, \
         open(out_path.joinpath("correct_nonzero_gold_train.src_tok"), "w") as gold_src_train_f, \
         open(out_path.joinpath("correct_nonzero_gold_test.src_tok"), "w") as gold_src_test_f:
        for correct_idx in correct_idxs_with_nonzero:
            gold_plan = correct_train_with_nonzero[min_correct_idxs[correct_idx]]
            pred_plan = correct_preds[correct_idx]
            gold_src = correct_train_src_with_nonzero[min_correct_idxs[correct_idx]]
            test_src = correct_gold_src[correct_idx]
            gold_src_train_f.write(f"{gold_src.strip()}\n")
            gold_src_test_f.write(f"{test_src.strip()}\n")
            gold_f.write(f"{' '.join(gold_plan)}\n")
            pred_f.write(f"{' '.join(pred_plan)}\n")

    with open(out_path.joinpath("incorrect_nonzero_gold.tgt"), "w") as gold_f, \
         open(out_path.joinpath("incorrect_nonzero_pred.tgt"), "w") as pred_f, \
         open(out_path.joinpath("incorrect_nonzero_gold_train.src_tok"), "w") as gold_src_train_f, \
         open(out_path.joinpath("incorrect_nonzero_gold_test.src_tok"), "w") as gold_src_test_f:

        for incorrect_idx in incorrect_idxs_with_nonzero:
            gold_plan = incorrect_train_with_nonzero[min_incorrect_idxs[incorrect_idx]]
            pred_plan = incorrect_preds[incorrect_idx]
            gold_src = incorrect_train_src_with_nonzero[min_incorrect_idxs[incorrect_idx]]
            test_src = incorrect_gold_src[incorrect_idx]
            gold_src_train_f.write(f"{gold_src.strip()}\n")
            gold_src_test_f.write(f"{test_src.strip()}\n")
            gold_f.write(f"{' '.join(gold_plan)}\n")
            pred_f.write(f"{' '.join(pred_plan)}\n")


    mean_correct = np.mean([min_correct[i] for i in correct_idxs_with_nonzero])
    mean_incorrect = np.mean([min_incorrect[i] for i in incorrect_idxs_with_nonzero])

    return mean_correct, mean_incorrect


def main(args):
    # quantify how much the model is memorizing by seeing if there is lower min levenshtein distance to train for examples that the model got correct 
    mean_correct, mean_incorrect = run_main(args.gold, args.pred, args.train, args.fxn_of_interest, args.anon)
    print(f"mean correct Levenshtein distance to train: {mean_correct:.2f}")
    print(f"mean incorrect Levenshtein distance to train: {mean_incorrect:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, help = "path to gold test .tgt file", required=True) 
    parser.add_argument("--pred", type=str, help = "path to predicted test .tgt file", required=True) 
    parser.add_argument("--train", type=str, help = "path to train .tgt file", required=True) 
    parser.add_argument("--fxn-of-interest", type=str, required=True) 
    parser.add_argument("--anon", action="store_true")
    args = parser.parse_args()
    main(args)