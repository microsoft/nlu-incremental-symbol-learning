# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

for dataset in "tree_dst"; do
    processed_text_data_dir="../resources/data/tree_dst.agent.data"
    mkdir -p "${processed_text_data_dir}"
    dataflow_dialogues_dir="../resources/data/tree_dst.agent.data" 
    for subset in  "train" "valid" "test"; do
        python -m dataflow.onmt_helpers.create_onmt_text_data \
            --dialogues_jsonl ${dataflow_dialogues_dir}/${subset}.dataflow_dialogues.jsonl \
            --num_context_turns 1 \
            --onmt_text_data_outbase ${processed_text_data_dir}/${subset} \
	    --include_agent_utterance \
            #--include_program \
            #--include_described_entities \
    done
done
