# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 

# Requires environment vars: 
# MODEL: the name of the model
# SEED: the random seed
# FXN: the number of the function 
# CHECKPOINT_ROOT: the dir to store all checkpoints

checkpoint_root="${CHECKPOINT_ROOT}/${MODEL}_${FACTOR}/${FXN}/${SEED}_seed"



for fxn_num in 15 30 75 
do
    FACTOR=$(echo "print(${fxn_num}/750)" | python) 
    echo ${FACTOR}
    for num in 750 1500 3000 7500 15000 18000 
    do
        checkpoint_dir="${checkpoint_root}/${num}_${fxn_num}"
        mkdir -p ${checkpoint_dir}
        echo "STARTING"
        python -u main.py \
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
            --upsample-constant-ratio ${FACTOR} \
            --upsample-constant-no-source \
            --source-triggers channel,radio,fm,point,station,tune \
            --device 0 | tee ${checkpoint_dir}/stdout.log 
    done
done


