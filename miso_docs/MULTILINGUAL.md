# Multilingual experiments

There are several multilingual experiments, which follow a pre-training and fine-tuning paradigm. 
There are two broad types of models we pre-train, which we need to fine-tune models later on.
In one case, we pretrain on UD data and finetune on UDS; in the other, we pre-train on UDS with varying architectures, and finetune on UD. 

## UD to UDS 
The steps to running these experiments are as follows:   

1. **Pretraining a UD model**:
To see how much having multilingual UD information benefits UDS parsing, we pretrain a multilingual UD model on 8 languages at once. 
The script to do this can be found in `miso/training_config/ud_experiments/multilingual/all_ud_transformer.jsonnet`

2. **Fine-tuning a UDS model**: 
Using the trained model from step 1, we can initialize a UDS parsing model. The configs to do this are in `miso/training_config/xlmr/transformer_pretrained/`, where the path to the model trained in (1) is given in the `pretrained_weights` field. 

## UDS to UD 
There are more models to train in this setting, as we train and evaluate on each language separately. 
The steps here are: 

1. **Pretraining UDS models**: We pretrain UDS models on joint EWT UD/UDS parsing as well as EWT UD parsing alone (to use as a control). 
The configs for this are in `miso/training_config/xlmr/transformer`. 

2. **Fine-tuning UD models**: For each architecture (syntax-only, encoder, intermediate) and each language, we fine-tune a separate UD model. The configs for doing this are in `miso/training_config/ud_experiments/monolingual/`. The `pretrained_weights` field should point to the `best.th` checkpoint file from step 1.

## Data organization
For AllenNLP to read the UD data properly, the following organziation is required: 

```
all_data
|
| train
    | 
    {lang_code}-universal.conllu
| dev 
    | 
    {lang_code}-universal.conllu
| test
    | 
    {lang_code}-universal.conllu
```

This data can be downloaded by running: 

```
wget https://nlp.jhu.edu/miso/data/ud/all_data.tar.gz
tar -xzvf all_data.tar.gz 
```


