"""
version 0.2

This script takes re-ordered output from lstm_decoder.lua or 
ngram_decoder.py and randomly replaces unk, UNK, and N (for numbers) symbols,
maintaining these symbols within base NPs (if applicable), using the original, 
un-preprocessed input.

None of the input files should have an EOS symbol.

    
"""


import string

import sys
import argparse
from collections import defaultdict
import random

random.seed(1776)


STACK_SEPARATOR = "<eos>"


NUMERIC_SYM = "N"

SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"



def token_contains_alpha(token):
    for c in token:
        if c.isalpha():
            return True
    return False
    
def token_contains_digit(token):
    for c in token:
        if c.isdigit():
            return True
    return False 
    
def token_contains_digit_and_no_alpha(token):
    return token_contains_digit(token) and not token_contains_alpha(token)
    

def get_word_groups(sentence_tokens):
    """
    Each sentence becomes a list of lists, where each item is either a base NP
    or a single token
    """
    
    word_groups = []
    seen_base_np = False
    word_group = []
    for token in sentence_tokens:
        if token == SONP_SYM:
            assert seen_base_np == False
            assert word_group == []
            seen_base_np = True
            word_group.append(token)
        elif token == EONP_SYM:
            assert seen_base_np != False
            assert word_group != []
            word_group.append(token)
            word_groups.append(word_group)
            word_group = []
            seen_base_np = False
        else:
            if seen_base_np:
                assert word_group != []
                word_group.append(token)
            else:
                assert word_group == []
                word_groups.append([token])
    return word_groups    
        
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--generated_reordering_with_unk', help="The generated word ordering file (optionally with base NP symbols). Does not contain EOS symbols.")
    parser.add_argument('-g', '--gold_unprocessed', help="The gold file without preprocessing (optionally with base NP symbols). Does not contain EOS symbols")
    parser.add_argument('-p', '--gold_processed', help="The gold file with preprocessing (optionally with base NP symbols). Does not contain EOS symbols")
    parser.add_argument('-o', '--out_file', help="Output path with filename of the genereated reordering recased and UNKs removed.")
    parser.add_argument('-n', '--remove_npsyms', help="Remove base NP symbols from --generated_reordering_with_unk.", action="store_true")
    
    
    args = parser.parse_args(arguments)
       
    generated_file = args.generated_reordering_with_unk
    gold_file = args.gold_unprocessed
    gold_processed_file = args.gold_processed
    out_file = args.out_file
    unk_sym = "unk"
    UNK_sym = "UNK"

    remove_npsyms = args.remove_npsyms

    generated_lines = []
    with open(generated_file) as f:
        for line in f:
            if line not in string.whitespace:
                line_split = line.split()
                assert line_split[-1] != STACK_SEPARATOR, "--generated_reordering_with_unk should not include End-of-sentence symbols"
                word_groups = get_word_groups(line_split)
                generated_lines.append(word_groups)
    
    gold_lines = []

    with open(gold_file) as f:
        for line in f:
            if line not in string.whitespace:
                line_split = line.split()
                assert line_split[-1] != STACK_SEPARATOR, "--gold_unprocessed should not include End-of-sentence symbols"
                word_groups = get_word_groups(line_split)
                gold_lines.append(word_groups)

    gold_processed_lines = []

    with open(gold_processed_file) as f:
        for line in f:
            if line not in string.whitespace:
                line_split = line.split()
                assert line_split[-1] != STACK_SEPARATOR, "--gold_processed should not include End-of-sentence symbols"
                word_groups = get_word_groups(line_split)               
                gold_processed_lines.append(word_groups)
                                
    assert len(generated_lines) == len(gold_lines) == len(gold_processed_lines)
    
    reprocessed_generated_sents = []

    for gen_sent, gold_sent, gold_proc_sent in zip(generated_lines, gold_lines, gold_processed_lines):
        assert len(gen_sent) == len(gold_sent) == len(gold_proc_sent)
        
        # dictionary that maps processed base NPs to gold base NPs (only 
        # processed base NPs with unk_sym or UNK_sym or the NUMERIC_SYM are included)
        processed_np_word_group_to_gold = defaultdict(list)
        # other processed groups:
        processed_word_group_to_gold = defaultdict(list)
        for gold_word_group, processed_word_group in zip(gold_sent, gold_proc_sent):
            assert len(gold_word_group) == len(processed_word_group)
            if len(processed_word_group) == 1:
                if (unk_sym in processed_word_group or UNK_sym in processed_word_group or NUMERIC_SYM in processed_word_group):
                    processed_word_group_to_gold[" ".join(processed_word_group)].append(gold_word_group)                
            else:
                assert processed_word_group[0] == SONP_SYM 
                if (unk_sym in processed_word_group or UNK_sym in processed_word_group or NUMERIC_SYM in processed_word_group):
                    processed_np_word_group_to_gold[" ".join(processed_word_group)].append(gold_word_group)

        reprocessed_generated_sent = []
        for generated_word_group in gen_sent:
            # each word group is a base NP or single token 
            if len(generated_word_group) == 1:
                if (unk_sym in generated_word_group or UNK_sym in generated_word_group or NUMERIC_SYM in generated_word_group):
                    possible_matches = processed_word_group_to_gold[" ".join(generated_word_group)]
                    idx = random.randint(0,len(possible_matches)-1)
                    reprocessed_generated_sent.append(" ".join(possible_matches[idx]))
                    del processed_word_group_to_gold[" ".join(generated_word_group)][idx]
                    if len(processed_word_group_to_gold[" ".join(generated_word_group)]) == 0:
                        del processed_word_group_to_gold[" ".join(generated_word_group)]
                else:
                    reprocessed_generated_sent.append(" ".join(generated_word_group))
            else:
                assert generated_word_group[0] == SONP_SYM
                if (unk_sym in generated_word_group or UNK_sym in generated_word_group or NUMERIC_SYM in generated_word_group):
                    possible_matches = processed_np_word_group_to_gold[" ".join(generated_word_group)]
                    idx = random.randint(0,len(possible_matches)-1)
                    reprocessed_generated_sent.append(" ".join(possible_matches[idx]))
                    del processed_np_word_group_to_gold[" ".join(generated_word_group)][idx]   
                    if len(processed_np_word_group_to_gold[" ".join(generated_word_group)]) == 0:
                        del processed_np_word_group_to_gold[" ".join(generated_word_group)]                 
                else:
                    reprocessed_generated_sent.append(" ".join(generated_word_group))
                    
        assert processed_word_group_to_gold == {} and processed_np_word_group_to_gold == {}
        
        reprocessed_generated_sent_no_npsyms = []
        if remove_npsyms:
            for word_group in reprocessed_generated_sent:
                for token in word_group.split():
                    if token not in [SONP_SYM, EONP_SYM]:
                        reprocessed_generated_sent_no_npsyms.append(token)
            reprocessed_generated_sent = reprocessed_generated_sent_no_npsyms
            
        reprocessed_generated_sent.append("\n")
        reprocessed_generated_sents.append(" ".join(reprocessed_generated_sent))
        
    with open(out_file, "w") as f:
        f.writelines(reprocessed_generated_sents) 
        
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    


