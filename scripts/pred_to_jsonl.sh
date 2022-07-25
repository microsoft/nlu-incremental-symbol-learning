# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

SPLIT=$1
onmt_text_data_dir="/brtx/601-nvme1/estengel/resources/data/smcalflow.agent.data" 
python -m dataflow.leaderboard.predict \
    --datum_id_jsonl ${onmt_text_data_dir}/${SPLIT}.datum_id \
    --src_txt ${onmt_text_data_dir}/${SPLIT}.src_tok \
    --ref_txt ${onmt_text_data_dir}/${SPLIT}.tgt \
    --nbest_txt ${CHECKPOINT_DIR}/translate_output/${SPLIT}.tgt \
    --nbest 1

mv predictions.jsonl ${CHECKPOINT_DIR}/translate_output/${SPLIT}_pred.jsonl
