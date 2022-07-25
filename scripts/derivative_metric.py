# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib 
import json 
import pdb 

import pandas as pd 
import numpy as np 


def get_calflow_from_path(path):
    return pd.read_csv(path).dropna(axis=0)

def get_intent_data_from_dir(root_data_dir, fxn, seeds):
    all_data = pd.DataFrame(columns=["train", "function", "seed", "total_acc", "fxn_acc"], dtype=object)
    root_data_dir = pathlib.Path(root_data_dir).joinpath(str(fxn))
    for seed in seeds:
        data_dir = root_data_dir.joinpath(f"{seed}_seed")

        globs = [x for x in data_dir.glob("*/test_metrics.json")]
        globs = sorted(globs, key = lambda x: int(x.parent.name.split("_")[0]))

        for path in globs:
            try:
                data = json.load(open(path))
            except json.JSONDecodeError:
                data = {}
                data['acc'] = np.nan
                data[f'{fxn}_acc'] = np.nan

            setting = path.parent.name
            num_train, num_fxn = setting.split("_")
            num_train, num_fxn = int(num_train), int(num_fxn)

            to_add = {"train": str(num_train), "function": num_fxn, "seed": seed, 
                     "total_acc": data['acc'], "fxn_acc": data[f"{fxn}_acc"]}
            all_data = all_data.append(to_add, ignore_index=True)

    return all_data 


def prepare_intent(df, fxn_num): 
    # drop total acc
    df = df.drop(columns="total_acc")
    # rename column 
    df = df.rename(columns={"fxn_acc":"acc"})
    df = df[df['function'] == fxn_num]
    # multiply accuracy by factor so that it is roughly the same size as train 
    df['train'] = df['train'].astype(int)
    avg_train = df.mean()['train']
    avg_acc = df.mean()['acc']
    ratio = avg_train / avg_acc
    df['acc'] *= ratio
    return df 


def prepare_calflow(df, fxn_num): 
    # drop total acc
    df = df.drop(columns="test_em")
    df = df.drop(columns="test_coarse")
    try:
        df = df.drop(columns="test_precision")
        df = df.drop(columns="test_recall")
        df = df.drop(columns="test_f1")
    except:
        pass 
    # rename column 
    df = df.rename(columns={"test_fine":"acc"})
    df = df[df['function'] == fxn_num]
    # multiply accuracy by factor so that it is roughly the same size as train 
    df['train'] = df['train'].astype(int)
    avg_train = df.mean()['train']
    avg_acc = df.mean()['acc']
    ratio = avg_train / avg_acc
    df['acc'] *= ratio
    return df 

def detect_missing(fxn, df, path, seeds=[12,31,64]):
    sub_dfs = []
    for seed in seeds:
        sub_df = df[df['seed'] == seed]
        sub_dfs.append(sub_df)

    for i,sdf_a in enumerate(sub_dfs):
        for j, sdf_b in enumerate(sub_dfs):
            if i == j:
                continue

            for train_setting in sdf_a['train']:
                b_vals = sdf_b[sdf_b['train'] == train_setting]
                if len(b_vals) == 0:
                    seed = sdf_a[sdf_a['train'] == train_setting]['seed'].values[0]
                    try:
                        fxn_num = sdf_a[sdf_a['train'] == train_setting]['fxn'].values[0]
                    except KeyError:
                        fxn_num = sdf_a[sdf_a['train'] == train_setting]['function'].values[0]
                    print(f"{path} {fxn}-{fxn_num} is missing {train_setting} for seed {seed}")
    
def compute_derivative_metric(df, average_first=False):
    # Assumed datatframe format: 
    # columns: "train", "function", "seed", "accuracy"
    # if average_first=True, then first take average across seeds, then take slope
    # otherwise take the average of slopes 
    timesteps = sorted(list(set([int(x) for x in df['train']])))
    slope_df = pd.DataFrame(columns=["end_train", "seed", "slope"], dtype=object)
    for i in range(len(timesteps)-1):
        start_ts = timesteps[i]
        end_ts = timesteps[i+1]

        start_rows = df[df['train'] == start_ts]
        end_rows = df[df['train'] == end_ts]

        # if average_first:
        #     start_acc = start_rows['acc'].mean()
        #     end_acc = end_rows['acc'].mean()
        #     slope = (end_acc - start_acc)/(end_ts - start_ts)
        # else:
        all_slopes = []
        for ((start_index, start_row), (end_index, end_row)) in zip(start_rows.iterrows(), end_rows.iterrows()):
            seed = start_row['seed']
            try:
                assert(start_row['seed'] == end_row['seed'])
                assert(int(start_row['train']) < int(end_row['train']))
                assert(start_row['function'] == end_row['function'])
            except AssertionError:
                pdb.set_trace() 
            start_acc = start_row['acc']
            end_acc = end_row['acc']
            single_slope = (end_acc - start_acc)/(end_ts - start_ts)
            all_slopes.append(single_slope)
            slope_df = slope_df.append({"end_train": end_ts, "seed": seed, "slope": single_slope}, ignore_index=True)
        #slope = np.mean(all_slopes)

    # pdb.set_trace()  
    sum_df = slope_df.set_index("seed").sum(level="seed")

    mean = sum_df.mean()['slope']
    stderr = sum_df.sem()['slope']

    print(f"mean: {mean} +/- {stderr}")
    return mean, stderr, sum_df

def prepare_latex(paths_and_settings, functions = [50, 66], seeds = [12, 31, 64], is_intent = True): 
    # columns: intent/function, #examples, setting, deriv_metric, min_acc, min_x, max_acc, max_x 
    latex_df = pd.DataFrame(columns=["function", "examples", "setting", "deriv_metric", "min_acc", "min_x", "max_acc", "max_x"],dtype=object)
    data_df = pd.DataFrame(columns=["function", "examples", "setting", "seed", "deriv_metric", "min_acc", "min_x", "max_acc", "max_x"],dtype=object)

    for path, setting in paths_and_settings:
        if is_intent:
            function, function_name, num_examples, model_name = setting

            df = get_intent_data_from_dir(path, function, seeds)
            acc_df = df[df['function'] == num_examples]
            prepped_df = prepare_intent(df, num_examples)
            key = "fxn_acc"
        else:
            function_name, num_examples, model_name, data_path = setting 
            df = get_calflow_from_path(path)
            key = "test_fine"
            max_len = get_max_len(data_path, function_name)
            acc_df = df[df['function'] == num_examples]
            acc_df['train'][acc_df['train'] == "max"] = max_len
            acc_df['train'] = acc_df['train'].astype(int)
            prepped_df = prepare_calflow(acc_df, num_examples)

        detect_missing(function_name, prepped_df, path)
        acc_df = acc_df.set_index("train").mean(level="train")
        mean, stddev, sum_df = compute_derivative_metric(prepped_df)
        metric_str = f"${mean:.2f}\pm{stddev:.2f}$"
        min_acc_idx = acc_df[key].argmin()
        min_acc_x = acc_df.index[min_acc_idx]
        min_acc_float = acc_df[key].min()
        min_acc = f"${min_acc_float}$"
        max_acc_idx = acc_df[key].argmax()
        max_acc_x = acc_df.index[max_acc_idx]

        if max_acc_x == "max":
            max_acc_x_int = max_len
        else:
            max_acc_x_int = int(max_acc_x)
        if min_acc_x == "max":
            min_acc_x_int = max_len
        else:
            min_acc_x_int = int(min_acc_x)

        max_acc_float = acc_df[key].max()
        max_acc = f"${max_acc_float}$"
        max_acc_x = f"${max_acc_x}$"
        min_acc_x = f"${min_acc_x}$"

        if type(max_acc_x) == int and max_acc_x > 100000:
           max_acc_x = "\\text{max}"
        max_acc_x = f"${max_acc_x}$"

        latex_df = latex_df.append({"function": function_name, "examples": num_examples, 
                                    "setting": model_name, "deriv_metric": metric_str, 
                                    "min_acc": min_acc, "min_x": min_acc_x,
                                    "max_acc": max_acc, "max_x": max_acc_x}, ignore_index=True)
        for row in sum_df.iterrows(): 
            # try:
            data_df = data_df.append({"function": function_name, "examples": num_examples, "seed": row[0], "setting": model_name, "deriv_metric": row[1]['slope'], 
                                "min_acc": min_acc_float, "min_x": min_acc_x_int, "max_acc": max_acc_float, "max_x": max_acc_x_int}, ignore_index=True) 
            # except KeyError:
                # pass 

    print(latex_df.to_latex(escape=False, float_format=".2f"))
    return data_df 

def get_max_len(data_path, function):
    data_path = pathlib.Path(data_path)
    with open(data_path.joinpath(function, "max_100", "train.src_tok")) as f1:
        data = f1.readlines()
    return len(data)


if __name__ == "__main__":

    intent = True
    if intent:
        paths = {"baseline": "/brtx/603-nvme1/estengel/intent_fixed_test/intent/",
                "remove source": "/brtx/603-nvme1/estengel/intent_fixed_test/intent_no_source",
                "upsample_16": "/brtx/606-nvme1/estengel/intent_fixed_test/intent_upsample_16.0",
                "upsample_32": "/brtx/606-nvme1/estengel/intent_fixed_test/intent_upsample_32.0"}
        functions_and_names = [ (50, "play_radio"), (66, "transit_traffic"), (15, "email_query"), (16, "email_querycontact"), (27, "general_quirky")]
        numbers = [15, 30, 75]

        paths_and_settings = []

        for num in numbers:
            for fxn, name in functions_and_names:
                for model_name, path in paths.items():
                    paths_and_settings.append((path, (fxn, name, num, model_name)))
    

    else:
        data_path = "/brtx/601-nvme1/estengel/resources/data/smcalflow_samples_curated/"
        paths = {"baseline": [("/home/estengel/papers/incremental_function_learning/results/DoNotConfirm_transformer_full_test.csv", "DoNotConfirm"),
                              ("/home/estengel/papers/incremental_function_learning/results/FindManager_transformer_test.csv", "FindManager"),
                              ("/home/estengel/papers/incremental_function_learning/results/Tomorrow_transformer_test.csv", "Tomorrow"),
                              ("/home/estengel/papers/incremental_function_learning/results/PlaceHasFeature_transformer_test.csv", "PlaceHasFeature")  
                             ]  
                }
        names = ["DoNotConfirm"]
        numbers = [100]

        paths_and_settings = []
        for num in numbers:
            for model_name, paths_and_fxns in paths.items():
                for (path, function_name) in paths_and_fxns: 
                    setting = (function_name, num, model_name, data_path)
                    paths_and_settings.append((path, setting))

    prepare_latex(paths_and_settings, is_intent=intent)
