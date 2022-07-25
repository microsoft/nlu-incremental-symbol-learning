# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv 
import argparse 
import re
import subprocess
import pathlib
import pdb 
import os 

import pandas as pd 


def collect_results(output, fxn):
    EM_GEX = re.compile("Exact Match: (\d+\.\d+)")
    COARSE_GEX = re.compile(f"{fxn} Coarse: (\d+\.\d+)") 
    FINE_GEX = re.compile(f"{fxn} Fine: (\d+\.\d+)") 
    PREC_GEX = re.compile(f"{fxn} Precision: -?(\d+\.\d+)") 
    REC_GEX = re.compile(f"{fxn} Recall: -?(\d+\.\d+)") 
    F1_GEX = re.compile(f"{fxn} F1: (\d+\.\d+)") 
    std_output = output[0].decode("utf-8")
    std_output = std_output.split("\n")[-7:]
    try:
        em_output = EM_GEX.search(std_output[0]).group(1)
        coarse_output = COARSE_GEX.search(std_output[1]).group(1)
        fine_output = FINE_GEX.search(std_output[2]).group(1)
        prec_output = PREC_GEX.search(std_output[3]).group(1)
        rec_output = REC_GEX.search(std_output[4]).group(1)
        f1_output = F1_GEX.search(std_output[5]).group(1)
    except AttributeError:
        print(f"exit with error {output[1]}")
        return None, None, None, None, None, None
    try:
        return float(em_output), float(coarse_output), float(fine_output), float(prec_output), float(rec_output), float(f1_output)
    except ValueError:
        return None, None, None, None, None, None

def get_results(model_dir, data_dir, fxn, test):
    print(f"getting results for {model_dir}...") 
    if test:
        prefix = "test"
    else:
        prefix = "dev"
    synthetic = False
    try:
        assert(model_dir.joinpath("translate_output", f"{prefix}_valid.tgt").exists())
    except AssertionError:
        try:
            assert(model_dir.joinpath("translate_output", f"{prefix}.tgt").exists())
            synthetic = True
        except AssertionError:
            print(f"Model dir {model_dir} has no output yet, skipping")
            return None, None, None, None, None, None

    #curr_path = pathlib.Path(__file__).resolve()
    #script_path = curr_path.parent.parent.joinpath("experiments","calflow.sh")
    if synthetic:
        script_path = "./experiments/calflow_synt.sh"
        test_data = str(data_dir.joinpath(f"{prefix}"))
    else:
        script_path = "./experiments/calflow.sh"
        test_data = str(data_dir.joinpath(f"{prefix}_valid"))
    my_env = os.environ.copy()
    my_env.update({"CHECKPOINT_DIR": str(model_dir), "TEST_DATA": test_data, "FXN": fxn})
    p = subprocess.Popen([str(script_path), "-a", "eval_pre"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env) 
    result = p.communicate() 
    return collect_results(result, fxn) 

def get_all_results(base_dir, data_dir, function = "FindManager", seeds = [12, 31, 64], splits = [5000, 10000, 20000, 50000, 100000, "max"], fxn_splits = [50, 100, 200, 500], test=False):
    if test:
        prefix = "test"
    else:
        prefix = "valid"

    colnames = ['train', 'function', 'seed', f'{prefix}_em', f'{prefix}_coarse', f'{prefix}_fine', 
                f'{prefix}_precision', f'{prefix}_recall', f'{prefix}_f1']
    df = pd.DataFrame(columns = colnames, dtype=object)

    for seed in seeds:
        for split in splits:
            for fxn_split in fxn_splits:
                model_dir = base_dir.joinpath(f"{function}_{seed}_seed", f"{split}_{fxn_split}")
                em, coarse, fine, precision, recall, f1 = get_results(model_dir, data_dir, function, test) 
                to_append = {"train": split, "function": fxn_split, "seed": seed, 
                            f"{prefix}_em": em, f"{prefix}_coarse": coarse, f"{prefix}_fine": fine,
                            f"{prefix}_precision": precision, f"{prefix}_recall": recall, f"{prefix}_f1": f1}
                df = df.append(to_append, ignore_index=True) 

    return df 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, help="path to dir where all models stored", required = True)
    parser.add_argument("--data-dir", type=str, help="path to dir where gold dev_valid.tgt stored", required = True)
    parser.add_argument("--out-path", type=str, help="path to write output csv", required = True) 
    parser.add_argument("--fxn", type=str, help="function to examine", default = "FindManager", required = False) 
    parser.add_argument("--splits", type=str, help = "training splits to examine (comma separated)", required = False, default = "5000,10000,20000,50000,100000,max")
    parser.add_argument("--fxn-splits", type=str, help = "function splits to examine (comma separated)", required = False, default = "50,100,200,500")
    parser.add_argument("--seeds", type=str, help="seeds to consider (comma separated)", required = False, default="12,31,64")
    parser.add_argument("--test", action="store_true") 
    args = parser.parse_args() 
    model_dir = pathlib.Path(args.model_dir)
    data_dir = pathlib.Path(args.data_dir)

    args.splits = [x.strip() for x in args.splits.split(",")]
    args.fxn_splits = [x.strip() for x in args.fxn_splits.split(",")]
    args.seeds = [x.strip() for x in args.seeds.split(",")]

    res_df = get_all_results(model_dir, data_dir, args.fxn, seeds = args.seeds, splits = args.splits, fxn_splits = args.fxn_splits, test=args.test) 
    with open(args.out_path,"w") as f1:
        res_df.to_csv(f1) 


