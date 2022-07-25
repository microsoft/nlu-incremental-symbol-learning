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
    #rm -rf ${CHECKPOINT_DIR}/ckpt
    echo "Training a new transductive model for decomp parsing..."
    python -um allennlp.run train \
    --include-package miso.data.dataset_readers \
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
    echo "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_parsing" \
    --batch-size 32 \
    --beam-size 1 \
    --use-dataset-reader \
    --save-pred-path ${CHECKPOINT_DIR}/${TEST_DATA}_graphs.pkl\
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  &> ${CHECKPOINT_DIR}/${TEST_DATA}.synt_struct.out
}

function eval_sem() {
    echo "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_parsing" \
    --batch-size 1 \
    --use-dataset-reader \
    --save-pred-path ${CHECKPOINT_DIR}/${TEST_DATA}_graphs.pkl\
    --semantics-only \
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  &> ${CHECKPOINT_DIR}/${TEST_DATA}.sem_struct.out

}

function eval_attr() {
    echo "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_parsing" \
    --include-attribute-scores \
    --batch-size 32 \
    --beam-size 1 \
    --use-dataset-reader \
    --save-pred-path ${CHECKPOINT_DIR}/${TEST_DATA}_graphs.pkl\
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics &> ${CHECKPOINT_DIR}/${TEST_DATA}.attr_struct.out
}

function spr_eval() {
    echo "Evaluating a transductive model for decomp parsing..."
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    mkdir ${CHECKPOINT_DIR}/${TEST_DATA} 
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score spr_eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_parsing" \
    --use-dataset-reader \
    --batch-size 32 \
    --oracle \
    --json-output-file ${CHECKPOINT_DIR}/${TEST_DATA}/data.json\
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics \
    --cuda-device 0 &> ${CHECKPOINT_DIR}/${TEST_DATA}.pearson.out
}

function conllu_eval() {
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    output_file=${CHECKPOINT_DIR}/${TEST_DATA}.pred.txt
    export PYTHONPATH=$(pwd)/miso:${PYTHONPATH}
    echo ${PYTHONPATH}
    python -m miso.commands.s_score conllu_eval \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_syntax_parsing" \
    --batch-size 1 \
    --beam-size 1 \
    --use-dataset-reader \
    --line-limit 128 \
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.predictors \
    --include-package miso.metrics
}

function serve() {
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    python -m miso.commands.predict predict \
    ${model_file} "NONE" \
    --predictor "decomp_syntax_parsing" \
    --batch-size 1 \
    --run-api \
    --use-dataset-reader \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.predictors \
    --include-package miso.metrics
}

function parse_lines() {
    model_file=${CHECKPOINT_DIR}/ckpt/model.tar.gz
    python -m miso.commands.predict predict \
    ${model_file} ${TEST_DATA} \
    --predictor "decomp_syntax_parsing" \
    --batch-size 1 \
    --run-arbitrary \
    --output-file ${CHECKPOINT_DIR}/output_file.pkl \
    --use-dataset-reader \
    --cuda-device 0 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
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
    elif [[ "${action}" == "spr_eval" ]]; then
        spr_eval
    elif [[ "${action}" == "eval_sem" ]]; then
        eval_sem 
    elif [[ "${action}" == "eval_attr" ]]; then
        eval_attr 
    elif [[ "${action}" == "conllu_eval" ]]; then
        conllu_eval
    elif [[ "${action}" == "serve" ]]; then
        serve
    elif [[ "${action}" == "parse_lines" ]]; then
        parse_lines
    fi
}


main "$@"
