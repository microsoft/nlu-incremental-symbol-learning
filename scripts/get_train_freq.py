# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import json 
import csv 
def get_train_freq(fxn):
    p = subprocess.Popen(['grep', '-c', fxn, '/home/t-eliass/resources/data/smcalflow.agent.data/train.tgt'], stdout=subprocess.PIPE) 
    out, err = p.communicate()
    return int(out.decode("utf-8")) 

if __name__ == "__main__":
    all_fxns = open('/home/t-eliass/scratch/fxn_names.txt').read().strip().split(",")
    existing = json.load(open('/home/t-eliass/papers/incremental_function_learning/results/acc_by_fxn.json'))


    for fxn in existing.keys():
        freq = get_train_freq(fxn)
        existing[fxn][1] = freq

    existing_items = sorted(existing.items(), key = lambda x: x[1][1])
    with open("/home/t-eliass/papers/incremental_function_learning/results/acc_by_train_freq.csv","w") as f1:
        writer = csv.writer(f1)
        writer.writerow(["function","acc", "freq"])
        for (fxn, (acc, freq)) in existing_items:
            writer.writerow([fxn, acc, freq])
        
    




