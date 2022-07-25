# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import numpy as np
import pdb 
import pathlib
import re 

from curate import curate
np.random.seed(12)

def read_data(basepath):
    src_path = basepath + ".src_tok"
    tgt_path = basepath + ".tgt"
    ids_path = basepath + ".datum_id"
    with open(src_path) as src_f, open(tgt_path) as tgt_f, open(ids_path) as ids_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()
        ids_lines = ids_f.readlines()
    assert(len(src_lines) == len(tgt_lines) == len(ids_lines))
    return list(zip(src_lines,tgt_lines, ids_lines))

def has_trigger(line, triggers): 
    splitline = re.split("[\s,\.\?]+", line.lower())
    for trig in triggers:
        if trig in splitline:
            return True
    return False 

def split_data(lines, fxn, exclude_triggers = None):
    if exclude_triggers is not None:
        exclude_triggers = exclude_triggers.split(",")
    interest_lines, rest_lines = [], []
    for src_line, tgt_line, id_line in lines:
        tgt_split = tgt_line.strip().split(" ")
        if fxn in tgt_split:
            interest_lines.append((src_line, tgt_line, id_line))
        else:
            if exclude_triggers is None:
                rest_lines.append((src_line, tgt_line, id_line))
            else:
                if not has_trigger(src_line, exclude_triggers):
                    rest_lines.append((src_line, tgt_line, id_line))
                else:
                    continue 
    if exclude_triggers:
        # make sure we excluded some examples 
        assert(len(lines) != len(rest_lines) + len(interest_lines))
    else:
        # make sure we covered the whole training set 
        assert(len(lines) == len(rest_lines) + len(interest_lines))

    return interest_lines, rest_lines 

def subsample(data, n_or_perc, max = False, perc = False, do_curate=False, read_path = None, write_path=None):
    # if not reading from file, recompute
    if read_path is None:
        if perc:
            total = len(data)
            n_or_perc = int(n_or_perc * total)
        try:
            if not do_curate:
                sample_idxs = np.random.choice([i for i in range(len(data))], size = n_or_perc, replace=False)
            else:
                sample_idxs = curate(data, size=n_or_perc)
        except ValueError:
            print(f"Warning: You asked for {n_or_perc} samples of a dataset with size {len(data)}, taking max amount: {len(data)}")
            sample_idxs = [i for i in range(len(data))]

        # if curating, make sure we're saving them 
        if write_path is not None:
            with open(write_path, "w") as f1:
                f1.write(",".join([str(x) for x in sample_idxs]))
    # read indices from file
    else:
        with open(read_path) as f1:
            sample_idxs = [int(x) for x in f1.read().split(",")]

    sample = [data[i] for i in sample_idxs]
    return list(sample) 

def write_data(data, basepath):
    bp = pathlib.Path(basepath)
    split = bp.name
    desired_num, sample_num = bp.parent.name.split("_")
    if int(desired_num) > len(data):
        new_name = f"max_{sample_num}"
        new_path = bp.parent.parent.joinpath(new_name).joinpath(split)
        new_path.mkdir(parents=True, exist_ok=True)
        basepath = str(new_path)

    src_path = basepath + ".src_tok"
    tgt_path = basepath + ".tgt"
    ids_path = basepath + ".datum_id"
    with open(src_path, "w") as src_f, open(tgt_path,"w") as tgt_f, open(ids_path, "w") as ids_f: 
        for src_line, tgt_line, id_line in data: 
            src_line = src_line.strip() + "\n"
            tgt_line = tgt_line.strip() + "\n"
            id_line = id_line.strip() + "\n"
            src_f.write(src_line)
            tgt_f.write(tgt_line)
            ids_f.write(id_line)

def main(args):
    train_data = read_data(args.train_path)
    # split into lines that contain fxn and those that don't 
    fxn_data, rest_data = split_data(train_data, args.fxn, exclude_triggers=args.exclude_triggers)
    if args.exact_n > -1:
        fxn_data_sample = subsample(fxn_data, args.exact_n, do_curate=args.do_curate, write_path=args.idx_write_path, read_path=args.idx_read_path)
    elif args.max_n > -1:
        fxn_data_sample = subsample(fxn_data, args.max_n, max=True)

    if args.total_perc > -0:
        rest_sample = subsample(rest_data, args.total_perc, perc=True)
    elif args.total_n > -0:
        rest_sample = subsample(rest_data, args.total_n - len(fxn_data_sample), perc=False)

    total_data = fxn_data_sample + rest_sample
    np.random.shuffle(total_data)
    #write_data(fxn_data_sample, args.out_path) 

    write_data(total_data, args.out_path) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, help="Path to full train data", required = True)
    parser.add_argument("--fxn", type=str, help = "type of function to sample", required = True)
    parser.add_argument("--max-n", type=int, help="maximum number of times fxn should appear", default = -1)
    parser.add_argument("--exact-n", type=int, help="exact number of times fxn should appear", default = -1)
    parser.add_argument("--total-n", type=int, help="total number of examples", default=-1)
    parser.add_argument("--total-perc", type=float, help="percentage of the original data to keep", default=-1.0)
    parser.add_argument("--out-path", type=str, help="path to output file", required=True)
    parser.add_argument("--do-curate", action="store_true", default=False)
    parser.add_argument("--exclude-triggers", type=str, default=None, help="comma-separated list of source triggers to exclude in building the rest of the dataset (e.g. boss,manager,skip for FindManager)") 
    parser.add_argument("--idx-write-path", type=str, help="path to save idxs if curating")
    parser.add_argument("--idx-read-path", type=str, help="path to read curated idxs " )
    args = parser.parse_args() 

    # check valid args 
    if (args.max_n > -1 and args.exact_n > -1) or \
        (args.max_n == -1 and args.exact_n == -1):
        raise AssertionError("exactly one of --max-n and --exact-n can be set")
    if (args.total_n > -1 and args.total_perc > -1.0) or\
        (args.total_n == -1 and args.total_perc == -1.0):
        raise AssertionError("exactly one of --total-n and --total-perc can be set")

    if args.do_curate:
        assert(args.idx_write_path is not None)

    main(args)
