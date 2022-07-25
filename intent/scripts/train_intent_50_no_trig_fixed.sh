# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 

# Requires environment vars: 
# MODEL: the name of the model
# SEED: the random seed
# CHECKPOINT_ROOT: the dir to store all checkpoints

FXN=50
checkpoint_root="${CHECKPOINT_ROOT}/${MODEL}/${FXN}/${SEED}_seed"

for num in 750 1500 3000 7500 15000 18000 
do
    for fxn_num in 15 30 75
    do
        checkpoint_dir="${checkpoint_root}/${num}_${fxn_num}"
        mkdir -p ${checkpoint_dir}
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u main.py \
            --data-path data/nlu_eval_data_${FXN}_no_source \
            --split-type interest \
            --bert-name bert-base-cased \
            --checkpoint-dir ${checkpoint_dir} \
            --batch-size 256 \
            --split-type interest \
            --total-train ${num} \
            --total-interest ${fxn_num} \
            --epochs 200 \
            --intent-of-interest ${FXN} \
            --seed ${SEED} \
            --device 0 | tee ${checkpoint_dir}/stdout.log 
    done
done


