# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pathlib
import argparse


from sample_functions import read_data, split_data

np.random.seed(12)


def write_data(data, basepath):
    bp = pathlib.Path(basepath)

    src_path = bp.joinpath("train.src_tok")
    tgt_path = bp.joinpath("train.tgt") 
    ids_path = bp.joinpath("train.datum_id") 
    with open(src_path, "w") as src_f, open(tgt_path,"w") as tgt_f, open(ids_path, "w") as ids_f: 
        for src_line, tgt_line, id_line in data: 
            src_line = src_line.strip() + "\n"
            tgt_line = tgt_line.strip() + "\n"
            id_line = id_line.strip() + "\n"
            src_f.write(src_line)
            tgt_f.write(tgt_line)
            ids_f.write(id_line)

def main(args):
    train_data = read_data(args.train)
    interest_lines, rest_lines = split_data(train_data, args.fxn_of_interest)
    if args.sample_rest:
        sample_n = args.num - len(rest_lines) 
        interest_idxs = [i for i in range(len(rest_lines))]
        sample_idxs  = np.random.choice(interest_idxs, size = sample_n, replace=True)
        samples = [rest_lines[idx] for idx in sample_idxs]
        train_data += samples
    else:
        sample_n = args.num - len(interest_lines) 
        interest_idxs = [i for i in range(len(interest_lines))]
        sample_idxs  = np.random.choice(interest_idxs, size = sample_n, replace=True)
        samples = [interest_lines[idx] for idx in sample_idxs]
        train_data += samples
    
    write_data(train_data, args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fxn-of-interest", required=True, type=str)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--sample-rest", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
