#from nltk.corpus import ptb
import nltk
from collections import defaultdict
import string
import glob
import numpy as np
import sys
import operator
import argparse

import random

random.seed(1776)


"""
version 1

This script takes a bracketed string (as from Gigaword) and inserts base NP
symbols. 

Following previous work, base noun phrases are defined as noun phrases without 
nested constituents.

"""



NUMERIC_SYM = "N"
LOW_COUNT_SYM = "<unk>"



SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"
NONE_NODE_LABEL = "-NONE-"



def subtree_contains_np(tree):
    """
    True if a subtree of tree contains an NP node; else, False
    """
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            #if subtree.label() == "NP":
            if subtree.label().startswith("NP"):
                return True
            else: # recurse further, depth-first
                if subtree_contains_np(subtree):
                    return True
        #else: # if leaf, then False
        #    return False
    return False
    

def filter_base_np(tree):
    """
    Exclude leaves with a parent NONE_NODE_LABEL
    """
    
    tokens = []
    if (type(tree) == nltk.tree.Tree) and (tree.label() == NONE_NODE_LABEL):
        return tokens
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == NONE_NODE_LABEL:
                return tokens
            else:
                child_tokens = filter_base_np(subtree)
                if len(child_tokens) > 0:
                    tokens.extend(child_tokens)
        else:
            tokens.append(str(subtree))
    return tokens
    
def traverse_tree(tree, return_flat_tree):
    """
    Traverse the tree, returning the leaves with base NP's marked with start (SONP_SYM) and end (EONP_SYM) tokens
    
    return_flat_tree: If False, base NP's are returned as nested lists
    """
    tokens = []
    if (type(tree) == nltk.tree.Tree) and (tree.label() == NONE_NODE_LABEL):
        return tokens
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            #if subtree.label() == "NP":
            if subtree.label().startswith("NP"):
                if tree.label() != NONE_NODE_LABEL:
                    # If the subtree does not contain additional NP's, the leaves constitute a 'base NP'
                    if not subtree_contains_np(subtree):
                        if subtree.leaves() and len(subtree.leaves()) > 0:  
                            # Filter the base-NP tree (as it may contain -NONE- labels):
                            filtered_leaves = filter_base_np(subtree)
    
                            if len(filtered_leaves) > 0:
                                if return_flat_tree:
                                    # use extend if a flat tree is desired:
                                    tokens.extend( [SONP_SYM] + filtered_leaves + [EONP_SYM])
                                else:
                                    # base-NPs will be in a nested list
                                    tokens.append( [SONP_SYM] + filtered_leaves + [EONP_SYM])
    
                        else:
                            assert(False)
                            
                    else:
                        # If the subtree contains NPs, continuing traversing in search of the base NP
                        child_tokens = traverse_tree(subtree, return_flat_tree)
                        if len(child_tokens) > 0:
                            tokens.extend(child_tokens)
            else:
                child_tokens = traverse_tree(subtree, return_flat_tree)
                if len(child_tokens) > 0:
                    tokens.extend(child_tokens)
                #tokens.append(traverse_tree(subtree))
        else:
            tokens.append(str(subtree))
    return tokens
    



def remove_base_np_syms(list_of_tokens_with_npsyms):
    """
    Remove the base NP symbols. No preprocessing/filtering is applied
        
    
    """
    
    tokens = []
    for token in list_of_tokens_with_npsyms:     
        if token != SONP_SYM and token != EONP_SYM:
            tokens.append(token)
    return tokens



def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_trees_file', help="A file containing bracketed trees, 1 per line, as produced by the Annotated Gigaword API and Command Line Tools v1.2.")
    parser.add_argument('-o', '--output_file', help="The output processed file. Contains <eos>. '.txt' will be added to end. \
        A second file (with the suffix '_no_npsyms.txt' on the name provided to --output_file) will contain words without base NP symbols. \
        Use this latter file to verify tree traversal against output from the AGIGA toolkit.")
    parser.add_argument('-p', "--print_every", type=int, default=10000, help="Print progress after processing this number of parse trees. (Default: 10000)")
                    
    
    args = parser.parse_args(arguments)
    input_trees_file = args.input_trees_file
    output_file = args.output_file

    print_every = args.print_every
    
    processed_sents = []
    processed_sents_no_npsyms = []
    line_ctr = 0
    with open(input_trees_file) as f:
        for line in f:
            if line not in string.whitespace:
                #if line_ctr > 6605:
                #    print line
                # In this version, convert non-ascii chars to string escape sequences
                tree_string = nltk.tree.Tree.fromstring(line.strip().decode('unicode_escape').encode("unicode_escape"))
                sent_with_npsyms = traverse_tree(tree_string, True)
                #print sent_with_npsyms
                processed_sents.append(" ".join(sent_with_npsyms) + "\n")
                #print remove_base_np_syms(sent_with_npsyms)
                processed_sents_no_npsyms.append(" ".join(remove_base_np_syms(sent_with_npsyms)) + "\n")
                if line_ctr % print_every == 0:
                    print "Finished processing sentence {LINE_CTR} in {INPUT_FILE}".format(LINE_CTR=line_ctr, INPUT_FILE=input_trees_file)
                line_ctr += 1
    with open(output_file+".txt", "w") as f:
        f.writelines(processed_sents)
    print "saved {OUTPUT_FILE}".format(OUTPUT_FILE=output_file+".txt")
    
    no_npsyms_output_file = output_file + "_no_npsyms.txt"
    with open(no_npsyms_output_file, "w") as f:
        f.writelines(processed_sents_no_npsyms)
    print "saved {OUTPUT_FILE}".format(OUTPUT_FILE=no_npsyms_output_file)

    print "Note that this output does not perform pre-processing on the ouput (such as for low frequency tokens), nor adds EOS."
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    






