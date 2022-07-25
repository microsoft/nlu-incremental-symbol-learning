# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

for seed in 12 31 64
do
    for num in 750 1500 3000 7500 15000 18000
    do 
        for fxn_num in 15 30 75
        do 
            python extract_difficult.py \
                --n-data ${num} \
                --n-intent ${fxn_num} \
                --out-path data/${num}_${fxn_num}_${seed}_seed.json \
                --intent 50 \
                --seed ${seed} 
        done
    done
done
