# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

FXN=$1
for seed in 12 31 64 
do
	for num in 5000 10000 20000 50000 100000 max 
	do
		for fnum in 50 200 500 
		#for fnum in 100
		do 
			python scripts/make_configs.py --base-jsonnet-config miso/training_config/calflow_transformer/FindManager/12_seed/5000_100.jsonnet --model-type transformer --function-type ${FXN} --json-out-path miso/training_config/calflow_transformer/ --data-split ${num}_${fnum} --seed ${seed}
		done
	done
done
