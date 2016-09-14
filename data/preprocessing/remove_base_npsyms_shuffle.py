import sys
import random

random.seed(1776)

"""
This script takes a file with base NPs and shuffles without regard to the 
base NPs.

"""


SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"

for l in sys.stdin:
    line = []
    for token in l.strip().split():
        if token not in [SONP_SYM, EONP_SYM, EOS_SYM]:
            line.append(token)

    random.shuffle(line)
    # add the EOS_SYM:
    line += [EOS_SYM]
    
    print " ".join(line)
