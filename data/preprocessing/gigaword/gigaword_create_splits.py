
import string
import glob
import numpy as np
import sys
import argparse
import os

import random

random.seed(1776)


"""

This script generates the sample from Gigaword. No processing/tokenization is performed.

"""



SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"


RETAIN_UNK_CASE = False

    

def main(arguments):

    #DATA_DIR="/n/rush_lab/users/schmaltz/projects/lm/gigaword_proc/stanford_phrase_structure_npsyms/"
    


    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_dir', help="Input directory with unprocessed files with NP symbols.")
    parser.add_argument('-o', '--output_splits_file', help="File containing 900k randomly selected AFP gigaword lines.")
    parser.add_argument('-k', '--output_splits_key_file', help="File containing file names and line numbers (indexed from 0 for each file) of contents of --output_splits_file.")    
    parser.add_argument('-n', '--sample_size', type=int, help="Sample size. (Default=900,000)", default=900000)           
    
    args = parser.parse_args(arguments)
    input_dir = args.input_dir
    output_splits_file = args.output_splits_file
    output_splits_key_file = args.output_splits_key_file
    sample_size = args.sample_size

    file_ctr = 0
    line_ctr = 0
    file_list = glob.glob(os.path.join(input_dir, "afp_eng_*.xml.gz_phrase_structure.txt_processed.txt"))
    #file_list = glob.glob(os.path.join(input_dir, "*.txt"))
    for one_file in file_list:
    #for one_file in glob.glob(input_dir + "*.txt"):
        file_ctr += 1
        print "processing %s (file number: %d)" % (one_file, file_ctr)
        with open(one_file) as f:
            for line in f:
                if line.strip() not in string.whitespace:
                    line_ctr += 1
    
    sorted_ids = sorted(random.sample(xrange(line_ctr), sample_size))
    
    samples_lines = []
    key_file = []
    
    file_ctr = 0
    sample_index = 0
    line_ctr = 0
    for one_file in file_list:
        file_ctr += 1
        print "sample from %s (file number: %d)" % (one_file, file_ctr)
        with open(one_file) as f:
            key_file.append(one_file + "\n")
            local_line_ctr = 0
            for line in f:
                line = line.strip()
                if line not in string.whitespace:
                    if sample_index < sample_size and line_ctr == sorted_ids[sample_index]:
                        sample_index += 1
                        key_file.append(str(local_line_ctr) + "\n")
                        
                        samples_lines.append(line + "\n")
                    local_line_ctr += 1
                    line_ctr += 1    
    
    print "Total lines: ", line_ctr
    
    with open(output_splits_file, "w") as f:
        f.writelines(samples_lines)
    print "saved {OUTPUT_FILE}".format(OUTPUT_FILE=output_splits_file)
    
    with open(output_splits_key_file, "w") as f:
        f.writelines(key_file)

    print "saved {OUTPUT_FILE}".format(OUTPUT_FILE=output_splits_key_file)


    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    






