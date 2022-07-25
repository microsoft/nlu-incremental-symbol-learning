# MISO for Universal Decompositional Semantic Parsing 

## What is MISO? 
MISO stands for Multimodal Inputs, Semantic Outputs. It is a deep learning framework with re-usable components for parsing a variety of semantic parsing formalisms. In various iterations, MISO has been used in the following publications: 

- [AMR Parsing as Sequence-to-Graph Transduction, Zhang et al., ACL 2019](https://www.aclweb.org/anthology/P19-1009/) 
- [Broad-Coverage Semantic Parsing as Transduction, Zhang et al., EMNLP 2019](https://www.aclweb.org/anthology/D19-1392/) 
- [Universal Decompositional Semantic Parsing, Stengel-Eskin et al. ACL 2020](https://www.aclweb.org/anthology/2020.acl-main.746/) 
- [Joint Universal Syntactic and Semantic Parsing, Stengel-Eskin et al., TACL 2021](#TODO) 

If you use the code in a publication, please do cite these works. 

## What is Universal Decompositional Semantics? 
[Universal Decompositional Semantics (UDS)](http://decomp.io/projects/decomp-toolkit/) is a flexible semantic formalism built on [English Web Treebank Universal Dependencies](https://universaldependencies.org/en/overview/introduction.html) parses. 
UDS graphs are directed acyclic graphs on top of UD parses which encode the predicate-argument structure of an utterance. 
These graphs are annotated with rich, scalar-valued semantic inferences obtained from human annotators via crowdsourcing, encoding speaker intuitions about a variety of semantic phenomena including factuality, genericity, and semantic proto-roles. 
More details about the dataset can be found in the following paper: [The Universal Decompositional Semantics Dataset and Decomp Toolkit, White et al., LREC 2020](https://www.aclweb.org/anthology/2020.lrec-1.699/) and at [decomp.io](http://decomp.io/projects/decomp-toolkit/). 

## What is UDS Parsing?  
UDS parsing is the task of transforming an utterance into a UDS graph, automatically. 
Using the existing dataset and the MISO framework, we can learn to parse into UDS. This is a particularly challenging parsing problem, as it involves three levels of parsing
1. Syntactic parsing of the utterance into UD 
2. Parsing the utterance into the UDS graph structure
3. Annotating the graph structure with UDS attributes

## MISO overview 
MISO builds heavily on [AllenNLP](https://github.com/allenai/allennlp), and so many of its core functionalities are the same. 

## Installation 
Using conda, all required libraries can be installed by running: 
- `conda create --name miso python=3.6`
- `conda activate miso`
- `pip install -r requirements.txt`

## Useful scripts 
`experiments/decomp_train.sh` has several functions for training and evaluating UDS parsers via the command-line. This script is used for `DecompParser` models, trained and evaluated without UD parses. Specifically:
- `train()` trains a new model from a configuration, saving checkpoints and logs to a directory specified by the user. If the directory is non-empty, an error will be thrown in order to not overwrite the current model.
- `resume()` is almost identical to `train` except that it takes a non-empty checkpoint directory and resumes training. 
- `eval()` evaluates the structure of the produced graphs with the S-metric, with unused syntax nodes included in the yield of each semantic head. It requires the user to specify which data split to use (test or dev). The outcome of the evaluation is stored in ${CHECKPOINT_DIR}/${TEST_DATA}.synt_struct.out.
- `eval_sem()` is identical to `eval` except that it computes the S-metric for semantics nodes only.
- `eval_attr()` evaluates the S-score with both syntactic nodes and attributes included. 
- `spr_eval()` computes the aggregate Pearson score across all nodes and edges, storing the outcome in ${CHECKPOINT_DIR}/${TEST_DATA}.pearson.out

If training/evaluating a model with syntactic info, a similar script is used: `syntax_experiments/decomp_train.sh`. This script has the same functions as `experiments/decomp_train`, but also contains: 
- `conllu_eval()`, which evaluates the micro-F1 LAS and UAS of predicted and gold UD parses
- `conllu_predict()` produces a CoNLL-U file with the predicted UD parse, which can be scored against a reference file using 3rd party tools. 
- `conllu_predict_multi()` which produces a similar file but from multilingual data (see [multilingual experiments](#TODO) for more)
- `conllu_predict_ai2()` produces a file for the PP-attachement dataset. 

## Configurations
For info on configurations, see [the configs page](docs/CONFIGS.md)

## Models 
For info on configurations, see [the models page](docs/MODELS.md)

## Training
For info on configurations, see [the training page](docs/TRAINING.md)

## Testing
For info on configurations, see [the testing page](docs/TESTING.md)
For predicting UDS graphs from arbitrary text, see [the prediction page](docs/UDS_PARSING.md)

## Multilingual experiments 
For info on configurations, see [the multilingual page](docs/MULTILINGUAL.md)
