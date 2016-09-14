"""

This script aligns base NP annotated sentences (dervied from the PTB
constituency trees) with the dependency trees, adding BNP annotations to the
dependency trees.

"""

from ptb_to_bnp_words import get_bnp_from_ptb
from collapse_dependency_trees_based_on_bnps import save_dependency_trees
from os import path
import argparse
import sys

  

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ptb_dir', help="Directory containing wsj")
    parser.add_argument('--data_dir', help="Data directory")
    
    args = parser.parse_args(arguments)
    
    ptb_dir = args.ptb_dir 
    data_dir = args.data_dir   
    
    dependency_dir = path.join(data_dir, "dependency")     
    filtered_dependency_dir = path.join(data_dir, "dependency_filtered_ordering_atomic")
     
    # get the BNP delineated splits:
    train_words, valid_words, test_words, train_bnps, valid_bnps, test_bnps = get_bnp_from_ptb(ptb_dir)
    
    # collapse BNPs:
    save_dependency_trees(train_words, valid_words, test_words, train_bnps, valid_bnps, test_bnps, dependency_dir, filtered_dependency_dir, True)
    
    # also save versions without BNPs:
    save_dependency_trees(train_words, valid_words, test_words, train_bnps, valid_bnps, test_bnps, dependency_dir, filtered_dependency_dir, False)    
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))