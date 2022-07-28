# Incremental Symbol Learning
Code and data for the paper: [When More Data Hurts: A Troubling Quirk in Developing Broad-Coverage Natural Language Understanding Systems](https://arxiv.org/abs/2205.12228) 


## Installation 

All dependencies can be installed with `./install_requirements.sh` 

## Downloading Data
The first step to replicating experiments is to download the data and glove embeddings.

From the project home directory:

```
mkdir -p data 
cd data
# This may take some time 
wget https://veliass.blob.core.windows.net/ifl-data/data_clean.tar.gz
tar -xzvf data_clean.tar.gz 
mv data_clean/* .
rm -r data_clean 
```

## Downloading models 
The models can be downloaded with the following command: 

```
wget https://veliass.blob.core.windows.net/ifl-models/models.tar.gz 
tar -xzvf models.tar.gz
```

The models distributed are the full dataset models reported in Table 1. The other models are too numerous to be distributed but can be replicated using the config files. 

## MISO model
The semantic parsing experiments in this paper use the MISO parser, which was developed across a series of papers: 

- [AMR Parsing as Sequence-to-Graph Transduction, Zhang et al., ACL 2019](https://www.aclweb.org/anthology/P19-1009/) 
- [Broad-Coverage Semantic Parsing as Transduction, Zhang et al., EMNLP 2019](https://www.aclweb.org/anthology/D19-1392/) 
- [Universal Decompositional Semantic Parsing, Stengel-Eskin et al. ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.746/) 
- [Joint Universal Syntactic and Semantic Parsing, Stengel-Eskin et al., TACL 2021](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00396/106796/Joint-Universal-Syntactic-and-Semantic-Parsing) 

More information on MISO can be found [here](MISO_README.md)

## File Organization 
Important directories: 
- `miso`: contains all the parsing code for the different MISO models 
- `scripts`: contains helper scripts for analysis and creating config files/data splits 
- `experiments`: contains bash files for running MISO parser (see MISO_README.md for more details) 

The main change between different `.jsonnet` files is the data path at the top. This points the model to the correct data split to use, e.g. `data/smcalflow_samples_curated/FindManager/5000_100/` 
points the model to the 5000 train sample subset with 100 FindManager examples. 
The assumption is that each experiment has a jsonnet file.
For example, the experiment which trains a transformer model with the `seed=12` for the 5000-100 FindManager corresponds to the `.jsonnet` file `miso/training_configs/calflow_transformer/FindManager/12_seed/5000_100.jsonnet`. 
In the released configs, the data dir argument is an environment variable 


## Important Scripts
- `scripts/sample_functions.py`: samples functions (e.g. FindManager) to create the different splits. Can be used to manually curate examples. 
- `scripts/make_subsamples.sh`: iteratively runs sampling for each split (5000-max), curating the first one and then using those examples later. 
- `scripts/make_subsamples_uncurated.sh`: same idea, but doesn't require curation (for non-100 splits, no curation is done).
- `scripts/make_configs.py`: can be used to modify a base jsonnet config to change the path to the split
- `scripts/prepare_data.sh`: Data is assumed to be pre-processed according to [Task Oriented Parsing as Dataflow Synthesis](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis) instructions. This is a modified version of the instructions in the README there to include agent utterances and previous user turns. 
- `scripts/collect_results.py`: script to collect exact match results from predictions, written to `CHECKPOINT_DIR/translate_output`. Aggregates all scores into a csv specified as an arg.
- `experiments/calflow.sh`: main training/testing commands for calflow

## Other scripts 
- `scripts/split_valid.py`: splits all valid dialogs into dev and test subsets. 
- `scripts/error_analysis.py`: for a given function, analyze predicted plans into 3 groups: correct predictions, incorrect examples wihtout the function, incorrect examples with the function. 
- `scripts/oversample.py`: either over-sample examples for a given function (e.g. turn 5000-100 FindManager into 5000-200 by doubling the 100 FindManager examples) or over-sample the rest of the training data to get a split of e.g. 200k-100 
where 200k is upsampled from the max setting. 

## Training Models 
Models can be trained locally using `experiments/calflow.sh`. 
`experiments/calflow.sh` expects the following environment variables to be set: `CHECKPOINT_DIR`, `TRAINING_CONFIG`, and `DATA_ROOT`. `DATA_ROOT` is the location where you downloaded the data. 
The former points to a directory where the model will store checkpoints. The latter is a `.jsonnet` config that will be read by AllenNLP. 
Optionally, the `FXN` variable can also be set, for function-specific evaluation. 

Model checkpoints and logs will be written to `CHECKPOINT_DIR/ckpt`. Decoded outputs will be written to `CHECKPOINT_DIR/translate_output/<split}>.tgt` 


For additional details, see [miso_docs/TRAINING.md](miso_docs/TRAINING.md) 

## Testing models 
The following environment variables need to set:
1. `CHECKPOINT_DIR`: the directory containing a subdirectory `ckpt`, which contains an archive `model.tar.gz`. If training is interrupted or canceled, the archive may be missing. It can be created manually by the following commands: 
```cd $CHECKPOINT_DIR/ckpt
cp best.th weights.th 
tar -czvf model.tar.gz weights.th config.json vocabulary
```
2. `TEST_DATA` is the path to the test data *without the extension*. An example would be `TEST_DATA=data/smcalflow.agent.data/dev_valid`. 
3. `FXN` is the function of interest. Example: `FXN=FindManager` 

The model can then be tested using `./experiments/calflow.sh -a eval_fxn`  

The output at the end will have the following rows: 

```
Exact Match: The overall exact match accuracy of produced and reference programs. 
FXN Coarse: The percentage of programs for which, if FXN is in the reference, it is also in the predicted program. It doesn't matter if the programs match or not. 
FindManager Fine: The percentage of programs with FXN in the reference where the predicted program is an exact match. 
FindManager Precision: The percentage of predicted programs that have FXN in them and also have FXN in the reference program. 
FindManager Recall: Same as Coarse 
FindManager F1: Harmonic mean of precision and recall 
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Legal Notices

Note that our data release is effectively a repartitioning of the SMCalFlow dataset and an English intent recognition dataset released by Liu et al. The SMCalFlow dataset
is available [here](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis) under a MIT license, which is available
[here](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis/blob/master/LICENSE). The intent recognition dataset is available [here](https://github.com/xliuhw/NLU-Evaluation-Data) under a CC BY 4.0 license, which is available [here](https://github.com/xliuhw/NLU-Evaluation-Data/blob/master/LICENSE).

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
