import sys

replace = {"<eos>": ""}

for l in sys.stdin:
    print " ".join([replace.get(w, w) for w in  l.strip().split()])
