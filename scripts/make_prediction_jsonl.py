# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json 
import pathlib 
import argparse 


def main(args):
    with open(args.data_path + ".datum_id") as id_file, open(args.pred_path) as pred_file:
        id_lines = id_file.readlines()
        pred_lines = pred_file.readlines()

    assert(len(id_lines) == len(pred_lines))
    output_jsonl = []
    
    with open(args.out_path, "w") as f1:
        for id_line, pred_line in zip(id_lines, pred_lines):
            jsonl = {"datum_id": json.loads(id_line), "lispress": pred_line.strip()} 
            f1.write(json.dumps(jsonl) + "\n") 




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/srv/local1/estengel/resources/data/smcalflow.agent.data/valid") 
    parser.add_argument("--pred-path", type=str, required=True) 
    parser.add_argument("--out-path", type=str, required=True) 
    args = parser.parse_args() 
    
    
    main(args) 


