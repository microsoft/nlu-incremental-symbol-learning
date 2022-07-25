# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pathlib
import json

np.random.seed(12) 

path = pathlib.Path("~/resources/data/smcalflow.agent.data") 

with open(path.joinpath("valid.dataflow_dialogues.jsonl")) as f1:
    json_data = f1.readlines()

n_dialogs = len(json_data) 

splitpoint = int(n_dialogs/2) 
np.random.shuffle(json_data)
dev = json_data[0:splitpoint]
test = json_data[splitpoint:]


with open(path.joinpath("dev_valid.dataflow_dialogues.jsonl"), "w") as dev_f, \
     open(path.joinpath("test_valid.dataflow_dialogues.jsonl"), "w") as test_f:
         for line in dev:
             dev_f.write(line.strip() + "\n")
         for line in test:
             test_f.write(line.strip() + "\n")


