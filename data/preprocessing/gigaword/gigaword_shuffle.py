
from collections import defaultdict
import string
import glob
import numpy as np
import sys
import argparse
from os import path
import random

random.seed(1776)


"""
This script shuffles the Gigaword validation and test datasets.
The resulting file is shuffled to match the shuffling of the 
datasets/zgen_data_npsyms_freq3_unkUNK datasets, but
removes unk/UNK symbols within the 25K Gigaword vocabulary.

This creates the applicable files for both the base NP and no-base NP files.

"""



NUMERIC_SYM = "N"
LOW_COUNT_SYM = "unk"
LOW_COUNT_SYM_UPPER = "UNK"



SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"


#TOTAL_VOCAB_SIZE = 25000


# all of the input files have eos

#wsj_valid_shuffled = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/valid_words_with_np_symbols_shuffled.txt"
#wsj_test_shuffled = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/test_words_with_np_symbols_shuffled.txt"
#
#wsj_valid = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/valid_words_with_np_symbols.txt"
#wsj_test = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/test_words_with_np_symbols.txt"
#
#gigaword_valid = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/valid_words_with_np_symbols.txt"
#gigaword_test = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/test_words_with_np_symbols.txt"
#
#gigaword_valid_shuffled = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/valid_words_with_np_symbols_shuffled.txt"
#gigaword_test_shuffled = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/test_words_with_np_symbols_shuffled.txt"


def get_lines_no_eos(input_file):
    """
    Return the non-blank lines of a file, removing the EOS_SYM from each line
    
    input_file File name of a text file containing one sentence per line ending
        in the EOS_SYM
    """
    
    lines = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line not in string.whitespace:
                line = line.split()
                assert line[-1] == EOS_SYM
                line = line[0:-1]
                lines.append(line)
    return lines


def get_lines(input_file):
    """
    Return the non-blank lines of a file
    """
    
    lines = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line not in string.whitespace:
                line = line.split()
                lines.append(line)
    return lines


def get_base_np_delineated_lines(lines):
    """    
    
    """
    
    base_np_delineated_lines = []
    for line in lines:
        collapsed_tokens = []
        running_np = []
        for token in line: # eos has already been excluded
            if token == SONP_SYM:
                assert len(running_np) == 0
                running_np.append(token)
            elif token == EONP_SYM:
                assert len(running_np) > 1
                running_np.append(token)
                collapsed_tokens.append(running_np)
                running_np = []                
            elif len(running_np) > 0:
                running_np.append(token)
            else:
                collapsed_tokens.append([token])        
        
        base_np_delineated_lines.append(collapsed_tokens)

    return base_np_delineated_lines

def save_file(file_name, list_of_lists):
    with open(file_name, "w") as f:
        f.writelines(list_of_lists)

    print "saved {OUTPUT_FILE}".format(OUTPUT_FILE=file_name)   

def main(arguments):

    #DATA_DIR="/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure_npsyms/"
    


    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zgen_data_npsyms_freq3_unkUNK_dir', help="Path to datasets/zgen_data_npsyms_freq3_unkUNK/")
    parser.add_argument('--gigaword_valid_out', help="The path to the PTB validation file with expanded Gigaword vocab (e.g., datasets/gigaword/valid_words_with_np_symbols.txt)")
    parser.add_argument('--gigaword_test_out', help="The path to the PTB test file with expanded Gigaword vocab (e.g., datasets/gigaword/test_words_with_np_symbols.txt)")
    parser.add_argument('--gigaword_shuffled_dir', help="The directory in which to save the shuffled BNP files")
    
    parser.add_argument('--gigaword_valid_out_no_npsyms', help="The path to the PTB validation file with expanded Gigaword vocab without BNP symbols (e.g., datasets/gigaword/valid_words.txt)")
    parser.add_argument('--gigaword_test_out_no_npsyms', help="The path to the PTB test file with expanded Gigaword vocab without BNP symbols (e.g., datasets/gigaword/test_words_with.txt)")
    parser.add_argument('--gigaword_no_npsyms_shuffled_dir', help="The directory in which to save the shuffled files without BNP symbols")    
    
    args = parser.parse_args(arguments)
    zgen_data_npsyms_freq3_unkUNK_dir = args.zgen_data_npsyms_freq3_unkUNK_dir    
    gigaword_valid_out = args.gigaword_valid_out        
    gigaword_test_out = args.gigaword_test_out
    gigaword_shuffled_dir = args.gigaword_shuffled_dir
    
    gigaword_valid_out_no_npsyms = args.gigaword_valid_out_no_npsyms        
    gigaword_test_out_no_npsyms = args.gigaword_test_out_no_npsyms
    gigaword_no_npsyms_shuffled_dir = args.gigaword_no_npsyms_shuffled_dir    
            
    for dataset_type in ["words", "npsyms"]:
        for split_name in ["valid", "test"]:
            wsj_file = path.join(zgen_data_npsyms_freq3_unkUNK_dir, "npsyms", "{SPLIT_NAME}_words_with_np_symbols_no_eos.txt".format(SPLIT_NAME=split_name))
            wsj_shuffled_file = path.join(zgen_data_npsyms_freq3_unkUNK_dir, "npsyms", "{SPLIT_NAME}_words_with_np_symbols_shuffled_no_eos.txt".format(SPLIT_NAME=split_name))
            if split_name == "valid":
                gigaword_file = gigaword_valid_out
            else:
                gigaword_file = gigaword_test_out

            gigaword_shuffled_file = path.join(gigaword_shuffled_dir, "{SPLIT_NAME}_words_with_np_symbols_shuffled.txt".format(SPLIT_NAME=split_name))
        
            if dataset_type == "words":
                wsj_file = path.join(zgen_data_npsyms_freq3_unkUNK_dir, "no_npsyms", "{SPLIT_NAME}_words_no_eos.txt".format(SPLIT_NAME=split_name))
                wsj_shuffled_file = path.join(zgen_data_npsyms_freq3_unkUNK_dir, "no_npsyms", "{SPLIT_NAME}_words_fullyshuffled_no_eos.txt".format(SPLIT_NAME=split_name))
                if split_name == "valid":
                    gigaword_file = gigaword_valid_out_no_npsyms
                else:
                    gigaword_file = gigaword_test_out_no_npsyms

                gigaword_shuffled_file = path.join(gigaword_no_npsyms_shuffled_dir, "{SPLIT_NAME}_words_fullyshuffled.txt".format(SPLIT_NAME=split_name))
                   
                    
    
    #for dataset_type in ["words", "npsyms"]:
    #    for split_name in ["valid", "test"]:
    #        wsj_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/{SPLIT_NAME}_words_with_np_symbols.txt".format(SPLIT_NAME=split_name)
    #        wsj_shuffled_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/{SPLIT_NAME}_words_with_np_symbols_shuffled.txt".format(SPLIT_NAME=split_name)
    #        gigaword_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/{SPLIT_NAME}_words_with_np_symbols.txt".format(SPLIT_NAME=split_name)
    #        gigaword_shuffled_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/{SPLIT_NAME}_words_with_np_symbols_shuffled.txt".format(SPLIT_NAME=split_name)
    #    
    #        if dataset_type == "words":
    #            wsj_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/no_npsyms/{SPLIT_NAME}_words.txt".format(SPLIT_NAME=split_name)
    #            wsj_shuffled_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/zgen_data_npsyms_freq3_unkUNK/no_npsyms/{SPLIT_NAME}_words_fullyshuffled.txt".format(SPLIT_NAME=split_name)
    #            gigaword_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/no_npsyms/{SPLIT_NAME}_words.txt".format(SPLIT_NAME=split_name)
    #            gigaword_shuffled_file = "/Users/a/Documents/MainC/LM/repo/lm/code/data/gigaword/no_npsyms/{SPLIT_NAME}_words_fullyshuffled.txt".format(SPLIT_NAME=split_name)           
            
            
            gigaword = get_base_np_delineated_lines(get_lines_no_eos(gigaword_file))
            wsj = get_base_np_delineated_lines(get_lines(wsj_file))
            wsj_shuffled = get_base_np_delineated_lines(get_lines(wsj_shuffled_file))
            
            
            
            wsj_to_gigaword = defaultdict(list)
            line_ctr = 0
            gigaword_shuffled = []
            for g_line, w_line in zip(gigaword, wsj):
                g_shuffled = []
                #print "g, w", g_line, w_line
                for g_phrase, w_phrase in zip(g_line, w_line):
                    #print "g, w", g_phrase, w_phrase
                    wsj_to_gigaword[" ".join(w_phrase)].append(" ".join(g_phrase))
                    
                for token in wsj_shuffled[line_ctr]:
            
                    token_str = " ".join(token)
                    possible_matches = wsj_to_gigaword[token_str]
                    #print possible_matches, token_str
                    idx = random.randint(0,len(possible_matches)-1)
                    g_shuffled.append(possible_matches[idx])
                    del wsj_to_gigaword[token_str][idx]
                    if len(wsj_to_gigaword[token_str]) == 0:
                        del wsj_to_gigaword[token_str]
                                    
                                    
                gigaword_shuffled.append(" ".join(g_shuffled) + " " + EOS_SYM + "\n")
                line_ctr += 1
        
            save_file(gigaword_shuffled_file, gigaword_shuffled)



    print "Complete"





if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


