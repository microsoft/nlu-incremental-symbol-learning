# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/bin/bash

#base="amlt_configs/transformer_no_source/"
base=$1
target=$2

for seed in 12 31 64
do
    for fxn in Tomorrow 
    do
        cp -r ${base}/FindManager_${seed}_seed  ${target}/${fxn}_${seed}_seed
        sed -i "s/FindManager/${fxn}/g" ${target}/${fxn}_${seed}_seed/*
    done
    for fxn in PlaceHasFeature FenceAttendee 
    do 
        cp -r ${base}/FindManager_${seed}_seed  ${target}/${fxn}_${seed}_seed
        sed -i "s/FindManager/${fxn}/g" ${target}/${fxn}_${seed}_seed/*
        sed -i "s/smcalflow_samples_curated/smcalflow_samples/g" ${target}/${fxn}_${seed}_seed/*

    done
    for fxn in DoNotConfirm 
    do
        cp -r ${base}/FindManager_${seed}_seed  ${target}/${fxn}_${seed}_seed
        sed -i "s/FindManager/${fxn}/g" ${target}/${fxn}_${seed}_seed/*
        sed -i "s/smcalflow_samples_curated/smcalflow_samples_big/g" ${target}/${fxn}_${seed}_seed/*

    done 
done
