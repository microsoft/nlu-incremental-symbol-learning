# Training 
Training a model can be done by using the script in experiments, or through AllenNLP (since all models here inherit from AllenNLP's `Model` class). For example, to train a joint UD-UDS transformer model with encoder-side biaffine parsing, we would use the following command: 

```
mkdir -p models/encoder 
python -um allennlp.run train \
    --include-package miso.data.dataset_readers \
    --include-package miso.data.tokenizers \
    --include-package miso.models \
    --include-package miso.modules.seq2seq_encoders \
    --include-package miso.training \
    --include-package miso.metrics \
    -s models/encoder/ckpt \
    miso/training_config/transformer/with_syntax/encoder.jsonnet
```

The weights, logs, and metrics will be saved to `models/encoder/ckpt`. 
The `--include-package` flags here tell AllenNLP where to look for the registered subclasses which are specified in the `encoder.jsonnet` config file. 
Various metrics and training progress will be logged to `stdout.log`. Any errors will appear in `stderr.log`.  
At the end of training, an archive called `model.tar.gz` containing the following files should be created: `weights.th`, `config.json`, `vocabulary/*`.
If training ends early for some reason and you want to evaluate the model anyway, you can create this file by accessing the model checkpoint directory and inputting 
```
cp best.th weights.th 
tar -czvf model.tar.gz weights.th config.json vocabulary
```