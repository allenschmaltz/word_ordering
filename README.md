# word_ordering
This repository includes code for replicating the results in the paper "Word Ordering Without Syntax" (2016).

The task of word ordering, or linearization, is to recover the original order of a shuffled sentence. It is an artificial, simplified task designed to isolate and compare certain aspects of generation models. The results of the experiments presented in our paper suggest that surface level models perform well (in terms of BLEU) on the standardized word ordering task compared to existing syntactic approaches, at least as currently implemented, ceteris paribus. This has potential implications for the utility of costly syntactic annotations in generation models more generally, for both high- and low- resource languages and domains.


Replicating our results can be broken down into two main steps:

1. Preprocess Penn Treebank with the splits and tokenization used in our experiments. Instructions are available in data/preprocessing/README_DATASET_CREATION.txt.

2. Train, run, and evaluate the NGram and LSTM models of interest. Instructions are available in Usage.txt



# Acknowledgements

We would like to that Jiangming Liu for pointing out a discrepancy (in calculating future costs of tokens that appear multiple times in a sentence) in the implementation of an earlier version of our NGram decoder, the resolution of which improved BLEU performance. The updated version appears in this repo.