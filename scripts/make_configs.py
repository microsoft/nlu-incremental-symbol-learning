# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml 
import re 
import argparse
import pathlib 
import pdb 

def read_base_jsonnet(path):
    with open(path) as f1:
        data = f1.read()
    return data 

def write_jsonnet(data, path):
    with open(path, "w") as f1:
        f1.write(data)
    print(f"wrote to {path}") 

def read_yaml(path):
    with open(path) as f1:
        #data = yaml.load(f1, loader=yaml.FullLoader) 
        data = yaml.load(f1) 
    return data

def write_yaml(data, path):
    with open(path, "w") as f1:
        yaml.dump(data, f1) 

def modify_jsonnet(data, function_type, data_split, seed): 
    #path_to_data = f"/mnt/default/resources/data/smcalflow_samples/{function_type}/{data_split}/"
    path_to_data = f"/mnt/default/resources/data/synthetic/{function_type}/{data_split}/"
    data = re.sub('local data_dir = ".*";', f'local data_dir = "{path_to_data}";', data)
    data = re.sub('seed: \d+', f'seed: {seed}', data)
    return data 

def modify_yaml(data, jsonnet_path, model_type, function_type, data_split): 
    checkpoint_dir= f"/mnt/output/{model_type}/{function_type}/{data_split}"
    for i in range(len(data['jobs'])):
        for j in range(len(data['jobs'][i]['command'])):
            if data['jobs'][i]['command'][j].strip().startswith("export CHECKPOINT_DIR"): 
                data['jobs'][i]['command'][j] = f'export CHECKPOINT_DIR={checkpoint_dir}'
            if data['jobs'][i]['command'][j].strip().startswith("export TRAINING_PATH"):
                data['jobs'][i]['command'][j] = f'export TRAINING_PATH={jsonnet_path}'

    return data 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--base-jsonnet-config", type=str, required=True)
    #parser.add_argument("--base-yaml-config", type=str, required=True)
    # parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--function-type", type=str, required=True)
    parser.add_argument("--data-split", type=str, required=True)
    parser.add_argument("--json-out-path", type=str, required=True)
    parser.add_argument("--seed", type=str, default = "12") 
    #parser.add_argument("--yaml-out-path", type=str, required=True) 
    args = parser.parse_args() 

    jsonnet_data = read_base_jsonnet(args.base_jsonnet_config)
    #yaml_data = read_yaml(args.base_yaml_config)
   
    new_jsonnet_data = modify_jsonnet(jsonnet_data, args.function_type, args.data_split, args.seed) 
    new_jsonnet_path = pathlib.Path(args.json_out_path).joinpath(args.function_type).joinpath(f"{args.seed}_seed").joinpath(args.data_split + ".jsonnet")
    write_jsonnet(new_jsonnet_data, new_jsonnet_path) 


