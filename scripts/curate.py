# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

np.random.seed(12)

def curate(data, size):
    sample_idxs = []
    data_idxs = [i for i in range(len(data))]
    while len(sample_idxs) < size:
        ex_idx = np.random.choice(data_idxs)
        datapoint = data[ex_idx]
        src_line, tgt_line, __ = datapoint
        print()
        print(f"src: {src_line.strip()}")
        print(f"tgt: {tgt_line.strip()}")
        code = input(f"good? [y]/N: ")
        if code in ['n','N','no','No']: 
            print("SKIPPED")
            continue
        else:
            print("ADDED")
            sample_idxs.append(ex_idx)
        print()
    return sample_idxs 

