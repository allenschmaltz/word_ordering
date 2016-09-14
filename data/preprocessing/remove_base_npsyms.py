import sys

SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"


for l in sys.stdin:
    line = []
    for token in l.strip().split():
        if token not in [SONP_SYM, EONP_SYM]:
            line.append(token)
    print " ".join(line)
