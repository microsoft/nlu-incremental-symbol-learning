
# Configurations 
MISO model instantiation happens via a config file, which is stored as a `.jsonnet` file. Config files are stored in `miso/training_config`, and are organized around 3 broad axes: 
- `lstm` vs `transformer` based models
- separate or joint models, where joint models do syntactic and semantic parsing jointly
- `semantics_only` or `with syntax` models, where `semantics only` models do not include non-head syntactic tokens in the yield of a semantic node, while `with_syntax` models do.  

Models also typically use GloVe embeddings, which need to be downloaded by the user and specified in the config file. 
The config files specify the registered names of classes, which are then instantiated with their options by AllenNLP. 
For example, a DecompDatasetReader is registered as a subclass of `allennlp.data.dataset_readers.dataset_reader` in `miso/data/dataset_readers/decomp.py` with the name `decomp` using the following line: 

```
@DatasetReader.register("decomp") 
```

and then instantiated from a config file with the following specification: 

```
"dataset_reader": {
    "type": "decomp",
    "drop_syntax": true,...
```

This method of setting and manipulating configuration options lets us save configurations easily and saves us from having excessively long commandline arguments. 
Furthermore, because the configs are `.jsonnet` files, we can set and re-use variables in them (e.g. paths to embeddings, etc.) 
The registration method used by allennlp and MISO also lets us easily extend classes to add new models and functionalities. 