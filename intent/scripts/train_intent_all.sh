# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 


# Requires environment vars: 
# MODEL: the name of the model
# SEED: the random seed
# FXN: the number of the function 
# CHECKPOINT_ROOT: the dir to store all checkpoints

checkpoint_root="${CHECKPOINT_ROOT}/${MODEL}/${FXN}/${SEED}_seed"

for fxn_num in 75 15 30
do
    for num in 750 1500 3000 7500 15000 18000 
    do
        echo "Visible: ${CUDA_VISIBLE_DEVICES}"
        checkpoint_dir="${checkpoint_root}/${num}_${fxn_num}"
        mkdir -p ${checkpoint_dir}
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -u main.py \
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


