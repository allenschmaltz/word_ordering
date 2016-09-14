"""
version 1

This scripts takes an output file from ZGen and converts it to sentences (one per
line, without EOS symbols) for use with the BLEU evaluation script.

Note that re-casing and UNK randomization are not currently performed.
    
"""


import string

import sys
import argparse


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input_zgen_file', help="Output from ZGen.")
    parser.add_argument('-o', '--output_file', help="Output file (1 sentence per line)")
    
    args = parser.parse_args(arguments)
       
    input_file = args.input_zgen_file
    output_file = args.output_file

    sentences = []
    sentence = []
    with open(input_file) as f:
        for line in f:
            if line not in string.whitespace:
                line = line.split("\t")
                tokens = line[0].split("__") # base NPs are marked with starting and trailing "__"
                if len(tokens) == 3: # base NP
                    assert tokens[0] == "" and tokens[2] == "", "ERROR: The following base NP is malformed: %s" % line[0]
                    sentence.extend(tokens[1].split("_")) # base NP constituent members are separated with "_"
                elif len(tokens) == 1: # not a base NP
                    sentence.append(tokens[0])
                else:
                    assert False, "ERROR: The following token string is malformed: %s" % line[0]
            else:
                # new sentence
                sentences.append(" ".join(sentence) + "\n")
                sentence = []
    if sentence != []:
        # in case the final sentence is missing a trailing blank line:
        sentences.append(" ".join(sentence) + "\n")
        sentence = []
            
    with open(output_file, "w") as f:
        f.writelines(sentences)  


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    


