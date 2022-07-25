# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib 
import re 

src_path, tgt_path = "/brtx/601-nvme1//estengel/resources/data/smcalflow.agent.data/test_valid.src_tok", "//brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data/test_valid.tgt" 

pred_path = "/home/estengel/papers/incremental_function_learning/results/no_source/DoNotConfirm_max_100_12_seed_test_valid.tgt"
difficult_pred = "/home/estengel/scratch/DoNotConfirm_max_100_12_seed_test_hard.tgt"

difficult_src, difficult_tgt = "/brtx/601-nvme1//estengel/resources/data/smcalflow.agent.data/DoNotConfirm_test_hard.src_tok", "//brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data/DoNotConfirm_test_hard.tgt" 


with open(src_path) as f1, open(tgt_path) as f2, open(pred_path) as f3: 
    src_data = f1.readlines() 
    tgt_data = f2.readlines() 
    pred_data = f3.readlines() 

og_src_data = [x for x in src_data]
og_tgt_data = [x for x in tgt_data]
for i, src_line in enumerate(src_data): 
    src_line = [x.lower() for x in re.split("\s+", src_line.split("__User")[-1]) ]
    src_data[i] = src_line 

tgt_data = [re.split("\s+", x) for x in tgt_data]

triggers = ["cancel", "n't", "no"]

hard_src, hard_tgt, hard_pred = [], [], []
for i, (src_l, tgt_l) in enumerate(zip(src_data, tgt_data)): 
    if "DoNotConfirm" in tgt_l:
        continue 
    for t in triggers: 
        if t in src_l: 
            print(i) 
            print(f"{' '.join(src_l)}")
            print(f"{' '.join(tgt_l)}")
            og_src_l = og_src_data[i]
            og_tgt_l = og_tgt_data[i]
            hard_pred.append(pred_data[i]) 
            hard_src.append(og_src_l) 
            hard_tgt.append(og_tgt_l) 

with open(difficult_src, "w") as f1, open(difficult_tgt, "w") as f2, open(difficult_pred, "w") as f3: 
    for src_l, tgt_l, pred_l in zip(hard_src, hard_tgt, hard_pred): 
        f1.write(src_l.strip() + "\n") 
        f2.write(tgt_l.strip() + "\n") 
        f3.write(pred_l.strip() + "\n") 
