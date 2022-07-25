# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash


for file in $(ls ~/amlt_models/synthetic_two_function_seq2seq/Func2_12_seed/) 
do
	export CHECKPOINT_DIR="/home/t-eliass/amlt_models/synthetic_two_function_seq2seq/Func2_12_seed/${file}"
	export TEST_DATA="/home/t-eliass/resources/data/synthetic_two_piece_balanced_same/seq2seq/Func2/${file}/test"
	export FXN="Func2"
	./experiments/calflow_synt.sh -a prob
done
