# UD Parsing Result

This work represents an improved parser for English Web Treebank UD parsing. 
The best-performing model is based on an XLM-Roberta encoder, which is partially fine-tuned first with multilingual UD parsing, and then trained to perform monolingual EWT UD parsing and, crucially, UDS semantic parsing. 
While obtaining SOTA results on UD parsing was by no means the objective of the project, these results might prove useful to researchers in the future, particularly because the same model can simultaneously perform UDS parsing in addition to UD parsing, unlike existing UD parsers.  
To that end, we have made the model available here.  It can be downloaded by running:

```
wget https://nlp.jhu.edu/miso/models/xlmr_intermediate_multilingual_pretraining/model.tar.gz
```

## Parsing all EWT data

Usually, the `DecompDatasetReader` object will skip any input graphs without a valid semantic parse. These typically are incomplete utterances like `(applause)` or URLs. 
However, to compare to other EWT UD parsing, we have decoded the full EWT dataset by simply assigning dummy semantic nodes to these invalid sentences. 
In order to run these experiments, the `DatasetReader` parameters need to include the `full_ud_parse` flag, as follows: 
```
"dataset_reader": {
          "type": "decomp_syntax_semantics",
          "drop_syntax": true,
          "full_ud_parse": true,
```

Once this flag is set, you will be able to run a UD decode as previously, using 

```
./syntax_experiments/decomp_train.sh -a decode_conllu -d {PATH_TO_MODEL} -i {dev,test}
```

## Parsing arbitrary sentences 
You can obtain UD parses for arbitrary sentences using the `conllu_predict_from_lines` functionality: 

```
./syntax_experiments/decomp_train.sh -a decode_conllu_from_lines -d {PATH_TO_MODEL} -i {PATH_TO_LINES_FILE}
```

where `PATH_TO_LINES_FILE` is a path to a file containing each sententence to parse and its comma-separated POS tags, separated by `\t`. 
For example, the first line of `en-ud-dev.lines` looks like this: 

```
From the AP comes this story :  ADP,DET,PROPN,VERB,DET,NOUN,PUNCT
```

POS tags are needed as input features to MISO. Such a file can be generated from a `.conllu` file using the conversion script in `scripts/make_ud_lines.py`. 
