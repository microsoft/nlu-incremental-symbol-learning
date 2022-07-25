## Intent recognition
In addition to semantic parsing, we explore intent recognition for the [nlu_evaluation_data](https://huggingface.co/datasets/nlu_evaluation_data) dataset. 

The intent recognition data is available via huggingface datasets. 

## Files
The following files are relevant to intent recognition: 

- `data.py` reads, loads, and splits the data 
- `main.py` runs the model, loading commandline arguments. For more info, please run `python main.py --help` 
- `model.py` contains the intent recognition model, which is just a classifier layer on top of BERT 
- `dro_loss.py` implements group DRO loss for intent recognition
- `extract_difficult.py` extracts difficult examples from the predicted and reference data for later analysis. 

## Configuration scripts
Each experiment is contained in a separate script, in the `scripts` dir. Scripts operate using environment variables, and each script has a docstring describing which environment variables need to be set. 

- `train_intent_upsample_fixed_ratio.sh` trains all models for an intent with the fixed ratio upsampling method.
- `train_intent_all.sh` runs over all intents at all resource settings and trains a model for that intent and setting. 
- `train_intent_group_dro.sh` runs group DRO training for all intents and resource settings 
- `train_intent_<num>_no_trig_fixed.sh` trains a model for each resource setting for intent <num> without the source-diluting trigger terms, using a fixed train set.
- `train_intent_upsample_constant_no_source_<num>.sh` trains a model with constant upsampling, but where diluting examples with the source triggers for <num> are excluded. This ensures that dilution remains the same. 
- `prepare_no_source/*` prepares the datasets for training for the no source dilution setting for each function. 

## Data
For reproducibility, the data (which can be loaded via Huggingface Datasets) is also duplicated in `data`. The organization is: 

- `data/nlu_eval_data`: the splits for all experiments except the "no source dilution" setting.
- `data/nlu_eval_data_<num>_no_source`: the splits for experiments without source dilution. Here, the train files have been modified to remove source-diluting examples. Since source-diluting examples are different for each intent, there is a separate split per intent. 


