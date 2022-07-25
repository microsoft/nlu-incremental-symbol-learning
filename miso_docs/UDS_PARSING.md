# UDS Parsing

In addition to predicting and evaluating EWT graphs using the code described [here](TESTING.md), you can predict UDS graphs from arbitrary sentences using the `experiments/decomp_train.sh:predict_lines()` function. 
This function takes as input a model checkpoint dir (as with the other testing functions) and a file which contains, line-by-line, the data you want to predict. 
It will write the graphs to a `.pkl` file, specified by the `--output-file` option, containing a list of UDSSentenceGraphs. 
