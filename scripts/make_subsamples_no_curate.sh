# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash
fxn=$1
in_path=$2
out_path=$3

for total_n in 5000 10000 20000 50000 100000 1200000
do
    for sample_n in 100 
    do
        output_dir="${out_path}/${fxn}/${total_n}_${sample_n}"
        mkdir -p ${output_dir}
        python ./scripts/sample_functions.py \
            --train-path ${in_path}/train \
            --out-path ${output_dir}/train \
            --fxn ${fxn} \
            --exact-n ${sample_n} \
            --total-n ${total_n} \
	    --idx-read-path data/${fxn}_${sample_n}_curated.idxs

        cp ${in_path}/dev_valid.src ${output_dir}
        cp ${in_path}/dev_valid.src_tok ${output_dir}
        cp ${in_path}/dev_valid.tgt ${output_dir}
        cp ${in_path}/test_valid.src ${output_dir}
        cp ${in_path}/test_valid.src_tok ${output_dir}
        cp ${in_path}/test_valid.tgt ${output_dir}
    done
done

for sample_n in 100 
do
    cp ${in_path}/dev_valid.* ${out_path}/${fxn}/max_${sample_n}
    cp ${in_path}/test_valid.* ${out_path}/${fxn}/max_${sample_n}
done
