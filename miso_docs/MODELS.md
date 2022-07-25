# Models 

### UDS Baselines 
- [DecompParser](https://github.com/esteng/miso_uds/blob/461a16ff1fd65cbfb034fa431b2412a128713ce4/miso/models/decomp_parser.py) is an LSTM-based UDS-only parsing model similar to the model presented [here](https://www.aclweb.org/anthology/2020.acl-main.746/). It has an encoder and a decoder. 
- [DecompTransformerParser](https://github.com/esteng/miso_uds/blob/461a16ff1fd65cbfb034fa431b2412a128713ce4/miso/models/decomp_transformer_parser.py) is a new Transformer-based variant of the `DecompParser` and inherits from it.  
### UD Baselines 
- [DecompSyntaxOnlyParser](https://github.com/esteng/miso_uds/blob/461a16ff1fd65cbfb034fa431b2412a128713ce4/miso/models/decomp_syntax_only_parser.py) is an LSTM-based biaffine parser, similar to the one presented [here](https://arxiv.org/abs/1611.01734). It only has an encoder (no decoder) and cannot do UDS parsing, just UD parsing.
- [DecompTransformerSyntaxOnlyParser](https://github.com/esteng/miso_uds/blob/master/miso/models/decomp_transformer_syntax_only_parser.py) the same model but with a Transformer encoder instead of an LSTM.
### Joint syntax-semantics models 
- [DecompSyntaxParser](https://github.com/esteng/miso_uds/blob/461a16ff1fd65cbfb034fa431b2412a128713ce4/miso/models/decomp_syntax_parser.py) can do joint UDS and UD parsing with an LSTM encoder/decoder, following one of three strategies: 
    - `concat-before` linearizes the UD parse and concatenates it before the linearized UDS graph
    - `concat-after` concatenates the UD tree after the UDS graph. Both the concatation strateiges are sub-optimal, since the UD formalism is lexicalized, so the model would need to learn to perfectly reconstruct the input tokens in a shuffled order, which is unnecessary. 
    - `encoder` puts a biaffine UD parser on top of the encoder in the encoder-decoder framework for UDS parsing. This model is similar to a `DecompSyntaxOnlyParser` but additionally has a decoder which performs UDS parsing. 
    - `intermediate` is almost the same as the `encoder` model, except that the output of the biaffine parser is re-encoded and passed to the decoder, making the decoder explicitly syntax-aware.  
- [DecompTransformerSyntaxParser](https://github.com/esteng/miso_uds/blob/461a16ff1fd65cbfb034fa431b2412a128713ce4/miso/models/decomp_transformer_syntax_parser.py) can do joint UDS and UD parsing with a Transformer encoder/decoder, following one of three strategies: 

### Transformer changes 
UDS has far fewer training examples than most tasks that transformers are applied to. Accordingly, we adopt a number of changes described in [Transformers without Tears: Improving the Normalization of Self-Attention](https://arxiv.org/abs/1910.05895) to adapt them to this low-data regime. 
These changes include:
- pre-normalization (swapping the order of the LayerNorm) 
- scaled initialization 
- smaller warmup rate 

### Contextualized encoders 
Currently, two different contextualized encoders can be used with MISO: BERT and XLM-Roberta (XLM-R). These are specified in config files under `bert_encoder`. Note that, if using a contextualized encoder, the appropriate tokenizer must also be set. BERT can be used by setting the `type` of the encoder to `seq2seq_bert_encoder`, while XLM-R encoders are registered under `seq2seq_xlmr_encoder`. 

