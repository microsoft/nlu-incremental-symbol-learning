# Testing 
Testing is a bit more complicated, since there are many metrics to consider. 

### Metrics 
#### S metric
This metric was designed for UDS graphs as an extension of Smatch. It can measure the alignment between a gold and predicted graph, including attribute scores. It is called via `miso.commands.s_score eval` 

```
    python -m miso.commands.s_score eval \
    models/encoder/ckpt/model.tar.gz dev \
    --predictor "decomp_syntax_parsing" \
    --batch-size 32 \
    --beam-size 1 \
    --use-dataset-reader \
    --save-pred-path models/encoder/pred_dev_graphs.pkl\
    --cuda-device -1 \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.predictors \
    --include-package miso.metrics  
```

As is, this version of S score is equivalent to Smatch. If attribute scores should be included in the overall score, the `--include-attribute-scores` flag should be set. 
If the `--semantics-only` flag is set, it will evaluate the score against just the semantics nodes, ignoring syntactic non-head nodes (see [Section 3](https://www.aclweb.org/anthology/2020.acl-main.746.pdf)).

#### Attribute Metrics
Under a forced ("oracle") decode of the graph structure, we can measure Pearson's rho between the predicted and gold attribute values for UDS node and edge attributes. 
This can be done by running `s_score.py` command with the `spr_eval` action, as done in the `spr_eval()` function in `syntax_experiments/decomp_train.sh`.
That command will produce a json file which has the predicted and gold edge and node attributes per node and edge.  
Running that json file through `python -m miso.commands.pearson_aggregate` will produce an average Pearson score for the whole system, averaged across all attributes. 

The script also produces an attribute F1 score. Recall that attributes are bounded on [-3, 3], typically with a midpoint at 0. We use the midpoint to binarize the values, computing the F1 score against the binarized gold attributes. In practice, we tune the threshold on the development set. 

#### UD Metrics
UD metrics used are unlabeled and labeled attachment score (UAS/LAS). These are standard UD metrics. 
The official metrics can be computed by running `miso/commands/s_score.py` with the `conllu_predict` argument as done in the `conllu_predict()` function in `syntax_experiments/decomp_train.sh`. 
This will produce a predicted `.conllu` file that can be evaluated against the gold file using the official evaluation script, which for convenience is included in `miso/metrics/conllu.py`. 
However, for debugging, a micro-averaged version of the conllu score can also be computed using the `conllu_eval()` function. 

#### UD data
EWT UD data is included in the repo in `data/UD/EWT` and `data/UD/EWT_clean`. The former directory has the official English UD data, while the latter has the filtered version described in the paper, where sentences without semantic graphs are filtered out. 

    