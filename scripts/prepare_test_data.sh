# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash 


data_path="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data/test.leaderboard_dialogues.jsonl"
subset="test" 
onmt_text_data_dir="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data"

python -m dataflow.leaderboard.create_text_data \
    --dialogues_jsonl ${data_path} \
    --num_context_turns 1 \
    --include_agent_utterance \
    --onmt_text_data_outbase ${onmt_text_data_dir}/${subset}
