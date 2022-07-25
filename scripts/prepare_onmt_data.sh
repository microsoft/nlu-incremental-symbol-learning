# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

onmt_data_stats_dir="../resources/data/onmt_data_stats"
mkdir -p "${onmt_data_stats_dir}"
python -m dataflow.onmt_helpers.compute_onmt_data_stats \
    --text_data_dir ${onmt_text_data_dir} \
    --suffix src src_tok tgt \
    --subset train valid \
    --outdir ${onmt_data_stats_dir}

onmt_binarized_data_dir="../resources/data/onmt_binarized_data"
mkdir -p "${onmt_binarized_data_dir}"

src_tok_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.src_tok.ntokens_stats.json)
tgt_max_ntokens=$(jq '."100"' ${onmt_data_stats_dir}/train.tgt.ntokens_stats.json)

# create OpenNMT binarized data
onmt_preprocess \
    --dynamic_dict \
    --train_src ${onmt_text_data_dir}/train.src_tok \
    --train_tgt ${onmt_text_data_dir}/train.tgt \
    --valid_src ${onmt_text_data_dir}/valid.src_tok \
    --valid_tgt ${onmt_text_data_dir}/valid.tgt \
    --src_seq_length ${src_tok_max_ntokens} \
    --tgt_seq_length ${tgt_max_ntokens} \
    --src_words_min_frequency 0 \
    --tgt_words_min_frequency 0 \
    --save_data ${onmt_binarized_data_dir}/data

# extract pretrained Glove 840B embeddings (https://nlp.stanford.edu/projects/glove/)
glove_840b_dir="../resources/glove_840b"
mkdir -p "${glove_840b_dir}"
unzip "../resources/glove.840B.300d.zip" -d ${glove_840b_dir}

onmt_embeddings_dir="../resources/data/onmt_embeddings"
mkdir -p "${onmt_embeddings_dir}"
python -m dataflow.onmt_helpers.embeddings_to_torch \
    -emb_file_both ${glove_840b_dir}/glove.840B.300d.txt \
    -dict_file ${onmt_binarized_data_dir}/data.vocab.pt \
    -output_file ${onmt_embeddings_dir}/embeddings
