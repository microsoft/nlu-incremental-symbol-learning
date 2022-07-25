
# Data

## Organization

For each function, there are varying splits across total number of training examples and total number of function examples (e.g. 5000_100 = 5000 training examples, 100 of which are the function). 
Each function will have it's own dir, with the splits as subdirs. Each subdir will have train, test, and dev files. 

## File types

- `XXX.src_tok`: tokenized user utterances with previous user utterance and agent utterance included, separated by special tokens (`__User` and `__Agent`) 
- `XXX.tgt`: tokenized linearized lispress sequences 
- `XXX.idx`: file of line indices to use in lookup later.

