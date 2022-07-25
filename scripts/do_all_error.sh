# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

FXN=$1

for seed in 12 31 64
do
	for num in 5000 10000 20000 50000 100000 max
	do
		for split in 100
		do
			mkdir -p ~/scratch/error_analysis/${FXN}/${seed}_seed/
			python scripts/error_analysis.py \
				--gold ~/resources/data/smcalflow.agent.data/test_valid.tgt \
				--input ~/resources/data/smcalflow.agent.data/test_valid.src_tok \
				--pred ~/amlt_models/transformer/${FXN}_${seed}_seed/${num}_${split}/translate_output/test_valid.tgt \
				--fxn-of-interest ${FXN} \
				--correct-output ~/scratch/error_analysis/${FXN}_min_pair_concat/${seed}_seed/${num}_${split}/correct \
				--incorrect-output-with-fxn ~/scratch/error_analysis/${FXN}_min_pair_concat/${seed}_seed/${num}_${split}/incorrect_with_fxn \
				--incorrect-output-without-fxn ~/scratch/error_analysis/${FXN}_min_pair_concat/${seed}_seed/${num}_${split}/incorrect_without_fxn 
		done
	done
done
