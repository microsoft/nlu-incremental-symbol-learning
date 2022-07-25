# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env bash

# Causes a pipeline (for example, curl -s http://sipb.mit.edu/ | grep foo)
# to produce a failure return code if any command errors
set -e
set -o pipefail

EXP_DIR=experiments
# Import utility functions.
#source ${EXP_DIR}/utils.sh

#CHECKPOINT_DIR=/exp/estengel/miso_res/models/decomp-parsing-ckpt
#TRAINING_CONFIG=miso/training_config/decomp_with_syntax.jsonnet
#TEST_DATA=dev


function train() {
    rm -rf ${CHECKPOINT_DIR}/ckpt
    echo "Training a new transductive model for decomp parsing..."

    python -um allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.training \
    --include-package miso.metrics \
    -s ${CHECKPOINT_DIR}/ckpt \
    ${TRAINING_CONFIG}
}

function resume() {
    python scripts/edit_config.py ${CHECKPOINT_DIR}/ckpt/config.json ${TRAINING_CONFIG}
    python -m allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.training \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.metrics \
    -s ${CHECKPOINT_DIR}/ckpt \
    --recover \
    ${TRAINING_CONFIG}
}


function test() {
    log_info "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/test.pred.txt
    python -m allennlp.run predict \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_parsing" \
    --batch-size 1 \
    --use-dataset-reader \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.predictors \
    --include-package miso.metrics
}


function eval() {
    echo "Evaluating Exact Match score for a transductive model for CalFlow parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    split=$(basename ${TEST_DATA})
    mkdir -p ${CHECKPOINT_DIR}/translate_output
    python -m miso.commands.exact_match eval \
    ${model_file} ${TEST_DATA} \
    --predictor "calflow_parsing" \
    --batch-size 400 \
    --beam-size 1 \
    --use-dataset-reader \
    --cuda-device 0 \
    --out-file ${CHECKPOINT_DIR}/translate_output/${split}.tgt \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
}

function eval_fxn() {
    echo "Evaluating Exact Match with Function Scores for Function ${FXN} for a transductive model for CalFlow parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    split=$(basename ${TEST_DATA})
    mkdir -p ${CHECKPOINT_DIR}/translate_output
    python -m miso.commands.exact_match eval \
    ${model_file} ${TEST_DATA} \
    --fxn-of-interest ${FXN} \
    --predictor "calflow_parsing" \
    --batch-size 140 \
    --beam-size 2 \
    --use-dataset-reader \
    --cuda-device 0 \
    --out-file ${CHECKPOINT_DIR}/translate_output/${split}.tgt \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
} 


function eval_fxn_cpu() {
    echo "Evaluating Exact Match with Function Scores for Function ${FXN} for a transductive model for CalFlow parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    split=$(basename ${TEST_DATA})
    mkdir -p ${CHECKPOINT_DIR}/translate_output
    python -m miso.commands.exact_match eval \
    ${model_file} ${TEST_DATA} \
    --fxn-of-interest ${FXN} \
    --predictor "calflow_parsing" \
    --batch-size 140 \
    --beam-size 2 \
    --use-dataset-reader \
    --cuda-device -1 \
    --out-file ${CHECKPOINT_DIR}/translate_output/${split}.tgt \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
} 

function eval_fxn_precomputed() {
    echo "Evaluating Exact Match with Function Scores for Function ${FXN} for a transductive model for CalFlow parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    split=$(basename ${TEST_DATA})
    python -m miso.commands.exact_match eval \
    ${model_file} ${TEST_DATA} \
    --fxn-of-interest ${FXN} \
    --predictor "calflow_parsing" \
    --batch-size 400 \
    --beam-size 2 \
    --use-dataset-reader \
    --cuda-device -1 \
    --precomputed \
    --out-file ${CHECKPOINT_DIR}/translate_output/${split}.tgt \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
} 

function prob_analysis() {
    echo "Evaluating Exact Match with Function Scores for Function ${FXN} for a transductive model for CalFlow parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    #output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    split=$(basename ${TEST_DATA})
    mkdir -p ${CHECKPOINT_DIR}/translate_output
    python -m miso.commands.exact_match eval \
    ${model_file} ${TEST_DATA} \
    --fxn-of-interest ${FXN} \
    --predictor "calflow_parsing" \
    --oracle \
    --score-type basic \
    --batch-size 1 \
    --beam-size 1 \
    --use-dataset-reader \
    --cuda-device 0 \
    --json-save-path ${CHECKPOINT_DIR}/translate_output/${split}_probs.json \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
} 

function big_beam() {
    echo "Evaluating Exact Match with Function Scores for Function ${FXN} for a transductive model for CalFlow parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    #output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    split=$(basename ${TEST_DATA})
    mkdir -p ${CHECKPOINT_DIR}/translate_output
    python -m miso.commands.exact_match eval \
    ${model_file} ${TEST_DATA} \
    --fxn-of-interest ${FXN} \
    --predictor "calflow_parsing" \
    --top-k-beam-search \
    --top-k 100 \
    --score-type basic \
    --batch-size 1  \
    --beam-size 100 \
    --use-dataset-reader \
    --cuda-device 0 \
    --out-file ${CHECKPOINT_DIR}/translate_output/${split}_top_100.tgt \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.iterators \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
} 

function usage() {

    echo -e 'usage: decomp_parsing.sh [-h] -a action'
    echo -e '  -a do [train|test|all].'
    echo -e "  -d checkpoint_dir (Default: ${CHECKPOINT_DIR})."
    echo -e "  -c training_config (Default: ${TRAINING_CONFIG})."
    echo -e "  -i test_data (Default: ${TEST_DATA})."
    echo -e 'optional arguments:'
    echo -e '  -h \t\t\tShow this help message and exit.'

    exit $1

}


function parse_arguments() {

    while getopts ":h:a:d:c:i:" OPTION
    do
        case ${OPTION} in
            h)
                usage 1
                ;;
            a)
                action=${OPTARG:='train'}
                ;;
            d)
                CHECKPOINT_DIR=${OPTARG:=${CHECKPOINT_DIR}}
                ;;
            c)
                TRAINING_CONFIG=${OPTARG:=${TRAINING_CONFIG}}
                ;;
            i)
                TEST_DATA=${OPTARG:=${TEST_DATA}}
                ;;
            ?)
                usage 1
                ;;
        esac
    done

    if [[ -z ${action} ]]; then
        echo ">> Action not provided"
        usage
        exit 1
    fi
}


function main() {

    parse_arguments "$@"
    if [[ "${action}" == "test" ]]; then
        test
    elif [[ "${action}" == "train" ]]; then
        train
    elif [[ "${action}" == "resume" ]]; then
        resume
    elif [[ "${action}" == "all" ]]; then
        train
        test
    elif [[ "${action}" == "eval" ]]; then
        eval
    elif [[ "${action}" == "eval_fxn" ]]; then
       eval_fxn 
    elif [[ "${action}" == "eval_fxn_cpu" ]]; then
       eval_fxn_cpu 
    elif [[ "${action}" == "eval_pre" ]]; then
       eval_fxn_precomputed 
    elif [[ "${action}" == "prob" ]]; then
      	prob_analysis 
    elif [[ "${action}" == "beam" ]]; then
      	big_beam 	
    fi
}


main "$@"
