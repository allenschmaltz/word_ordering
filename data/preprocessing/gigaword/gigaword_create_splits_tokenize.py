
from collections import defaultdict
import string
import glob
import numpy as np
import sys
import operator
import argparse
import os

import random

random.seed(1776)


"""
This script processes the sample from Gigaword, adding the WSJ training, 
and updating the valid and test WSJ splits with the new vocab. A separate
script is needed for shuffling (to match existing files).

(Note that this file adds EOS symbols to the end of files. These are not 
expected in the current input format for the decoders, so they need to be 
removed, as noted in the README.)

"""



NUMERIC_SYM = "N"
LOW_COUNT_SYM = "unk"
LOW_COUNT_SYM_UPPER = "UNK"



SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"


TOTAL_VOCAB_SIZE = 25000


def numeric_preprocess(token):
    for c in token:
        if c.isdigit():
            token = NUMERIC_SYM
            break
    return token

def preprocess_token(token, strip_low_freq, types_retained):
    """
        
    """
    
    # as an initial rough transformation, convert to N any token with a digit (may include parts of prices, dates, numbers with comma separators, fractions, etc.)
    for c in token:
        if c.isdigit():
            token = NUMERIC_SYM
            break

    if strip_low_freq and types_retained:
        if token not in types_retained:
            if token[0].isupper():
                token = LOW_COUNT_SYM_UPPER
            else:
                token = LOW_COUNT_SYM

    return token
    
def get_preprocessed_lines_from_file(file_name, word_types_retained):
    processed_lines = []
    line_ctr = 0
    with open(file_name) as f:
        for line in f:
            sent = []
            line = line.strip().split()
            for token in line:
                token = preprocess_token(token, True, word_types_retained)
                sent.append(token)
            sent.append(EOS_SYM)
            processed_lines.append(" ".join(sent) + "\n")
            line_ctr += 1
            if line_ctr % 10000 == 0:
                print "Currently processing tokens in line ", line_ctr   
    return processed_lines 

def save_file(file_name, list_of_lists):
    with open(file_name, "w") as f:
        f.writelines(list_of_lists)

    print "saved {OUTPUT_FILE}".format(OUTPUT_FILE=file_name)   
    
    
def main(arguments):

    #DATA_DIR="/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure_npsyms/"
    


    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gigaword_file', help="Sample created by the gigaword_create_splits.py script (e.g., gigaword/sample/afp_900k_sample.txt)")
    parser.add_argument('--wsj_train_file', help="The path to the file datasets/zgen_data_npsyms_freq3_unkUNK/npsyms/train_words_with_np_symbols_no_eos.txt")
         
    parser.add_argument('--wsj_train_gold', help="The path to the file datasets/zgen_data_gold/train_words_ref_npsyms.txt")
    parser.add_argument('--wsj_valid_gold', help="The path to the file datasets/zgen_data_gold/valid_words_ref_npsyms.txt")
    parser.add_argument('--wsj_test_gold', help="The path to the file datasets/zgen_data_gold/test_words_ref_npsyms.txt")
    
    parser.add_argument('--train_out', help="The path to the file in which to save the Gigaword+PTB training file (e.g., datasets/gigaword/sample/afp_900k_sample_with_wsjtrain_processed.txt)")
    parser.add_argument('--valid_out', help="The path to the file in which to save the updated PTB validation file (e.g., datasets/gigaword/valid_words_with_np_symbols.txt)")
    parser.add_argument('--test_out', help="The path to the file in which to save the updated PTB test file (e.g., datasets/gigaword/test_words_with_np_symbols.txt)")
    
    args = parser.parse_args(arguments)
    gigaword_file = args.gigaword_file
    wsj_train_file = args.wsj_train_file

    wsj_train_gold = args.wsj_train_gold
    wsj_valid_gold = args.wsj_valid_gold
    wsj_test_gold = args.wsj_test_gold
    
    train_out = args.train_out
    valid_out = args.valid_out
    test_out = args.test_out
        
   
    ## none of the input files have eos
    #gigaword_file = "/Users/a/Downloads/out/gigaword/sample/afp_900k_sample.txt"
    #wsj_train_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/no_eos/train_words_with_np_symbols_no_eos.txt"
    #
    #
    #wsj_train_gold = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_gold/train_words_ref_npsyms.txt"
    #wsj_valid_gold = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_gold/valid_words_ref_npsyms.txt"
    #wsj_test_gold = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_gold/test_words_ref_npsyms.txt"
    #
    #train_out = "/Users/a/Downloads/out/gigaword/sample/afp_900k_sample_with_wsjtrain_processed.txt"
    #valid_out = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/valid_words_with_np_symbols.txt"
    #test_out = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/test_words_with_np_symbols.txt"
    
    word_types_retained = {}
    with open(wsj_train_file) as f:
        for line in f:
            line = line.strip().split()
            for token in line:
                if token not in word_types_retained:
                    word_types_retained[token] = 1
                else:
                    word_types_retained[token] += 1
    
    total_vocab = len(word_types_retained)
    print "Tokens in WSJ processed train:", total_vocab     
    
    
    line_ctr = 0
    types_to_freq = defaultdict(int)
    with open(gigaword_file) as f:
        for line in f:
            line = line.strip().split()
            for token in line:
                numeric_processed_token = numeric_preprocess(token)
                types_to_freq[numeric_processed_token] += 1 
            line_ctr += 1
            if line_ctr % 100000 == 0:
                print "Currently processing line ", line_ctr
    
    # sort gigaword vocab:
    
    word_freq_sorted = sorted(types_to_freq.items(), key=lambda x: x[1], reverse=True)        
    
    # add additional word types from Gigaword up to TOTAL_VOCAB_SIZE
    for word_type, word_freq in word_freq_sorted:
        if word_type not in word_types_retained:
            word_types_retained[word_type] = word_freq
            total_vocab += 1
        else:
            word_types_retained[word_type] += word_freq
            
        if total_vocab >= TOTAL_VOCAB_SIZE:
            break
    
    print "Total number of types retained:", len(word_types_retained)
    
    
    
    
    # preprocess all files (including adding eos)
    gigaword_lines = get_preprocessed_lines_from_file(gigaword_file, word_types_retained)
    train_lines = get_preprocessed_lines_from_file(wsj_train_gold, word_types_retained)
    valid_lines = get_preprocessed_lines_from_file(wsj_valid_gold, word_types_retained)
    test_lines = get_preprocessed_lines_from_file(wsj_test_gold, word_types_retained)                            
    
    # now, concatentate WSJ train with the AFP sample
    gigaword_lines.extend(train_lines)
    
    print "Finished processing. Now saving."
    save_file(train_out, gigaword_lines)
    save_file(valid_out, valid_lines)
    save_file(test_out, test_lines)
    
    print "Complete"
    



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))



