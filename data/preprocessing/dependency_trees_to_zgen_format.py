"""
This one-off script converts base NP, projected CoNLL data to the format
used by ZGen and the Yara parser.

This version outputs unfiltered files, with casing and low frequency words retained.

This version attempts to match the format suggested by Jiangming Liu (email 
communication).

Note that ZGen expects a different format for arc/head numbering than the standard
CoNLL format. Namely, ZGen expects the ROOT index to be -1. It is sufficient to
just decrement the CoNLL indecies by 1. (See the constant ZGEN_OFFSET below.)

"""

import string
import sys
import argparse
from os import path
from ptb_to_bnp_words import get_bnp_from_ptb
from collections import defaultdict

BASE_NP_SEPARATOR = "^^"
SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"
ZGEN_OFFSET = 1

def get_reformatted_sents_yara(input_filename, base_np_sentences, constraints, unfiltered_sentences):
    
    token_ctr = 0
    sent_ctr = 0
    unfilt_sent_ctr = 0
    reformatted_sents = []
    reformatted_sent = []
    token_id_ctr = 1
    with open(input_filename) as f:
        for line in f:
            line = line.strip()
            if line not in string.whitespace:
                line = line.split("\t")
                tokens = line[1].split(BASE_NP_SEPARATOR)
                reformatted_tokens = []
                reformatted_token_string = ""
                seen_basenp = False
                for token in tokens:
                    if base_np_sentences[sent_ctr][token_ctr] != token:
                        assert base_np_sentences[sent_ctr][token_ctr] == SONP_SYM
                        assert len(reformatted_tokens) == 0
                        reformatted_tokens.append("_")
                        assert base_np_sentences[sent_ctr][token_ctr+1] == token
                        #reformatted_tokens.append(token)
                        #print sent_ctr, unfilt_sent_ctr
                        reformatted_tokens.append(unfiltered_sentences[sent_ctr][unfilt_sent_ctr])
                        token_ctr += 1
                        seen_basenp = True
                    else:
                        #reformatted_tokens.append(token)
                        reformatted_tokens.append(unfiltered_sentences[sent_ctr][unfilt_sent_ctr])
                    unfilt_sent_ctr += 1
                    token_ctr += 1
                assert len(reformatted_tokens) >= 1
                if len(reformatted_tokens) > 1: # have seen a base NP and need to end EONP_SYM
                    assert base_np_sentences[sent_ctr][token_ctr] == EONP_SYM
                    assert seen_basenp
                    token_ctr += 1
                    reformatted_tokens.append("_")
                    reformatted_token_string = "_".join(reformatted_tokens)
                else:
                    reformatted_token_string = reformatted_tokens[0]
                #print reformatted_token_string
                pos_tag = line[3]
                if constraints == "YaraDep":
                    head_id = str(int(line[6]) - ZGEN_OFFSET + 1)
                else:
                    head_id = str(int(line[6]) - ZGEN_OFFSET)
                deprel = line[7]
                reformatted_row = []
                if constraints == "YaraDep":
                    reformatted_row.append(str(token_id_ctr))
                token_id_ctr += 1
                reformatted_row.append(reformatted_token_string)
                if "__" in reformatted_token_string:
                    pos_tag = "NP"
                #if CONSTANT_DEP_LABEL:
                #    deprel = "N"
                if constraints == "Ref":
                    reformatted_row.extend([pos_tag, head_id, deprel])
                elif constraints == "OnlyPos":
                    reformatted_row.extend([pos_tag, "-1", "_"])
                elif constraints == "ArcsPos":
                    reformatted_row.extend([pos_tag, head_id, "_"])
                elif constraints == "OnlyArcs":
                    reformatted_row.extend(["-NONE-", head_id, "-NONE-"])
                elif constraints == "Input":
                    reformatted_row.extend(["-NONE-", "-1", "-NONE-"])
                elif constraints == "YaraDep":
                    reformatted_row.extend(["_", pos_tag, "_", "_", head_id, deprel, "_", "_"]) 
                elif constraints == "YaraPos":
                    reformatted_row[-1] = reformatted_row[-1] + "_" + pos_tag
                elif constraints == "ZparTok":
                    assert len(reformatted_row) == 1
                    
                if constraints == "YaraPos" or constraints == "ZparTok":
                    reformatted_sent.append(" ".join(reformatted_row))
                else:
                    reformatted_sent.append("\t".join(reformatted_row))
            else:
                token_ctr = 0
                token_id_ctr = 1
                unfilt_sent_ctr = 0
                sent_ctr += 1
                assert len(reformatted_sent) > 0
                if constraints == "YaraPos" or constraints == "ZparTok":
                    reformatted_sents.append(" ".join(reformatted_sent) + "\n")
                else:
                    reformatted_sents.append("\n".join(reformatted_sent) + "\n")
                    reformatted_sents.append("\n")
                reformatted_sent = []
    
    # check that the final sentence was added
    assert reformatted_sent == []
    return reformatted_sents
    
def get_reformatted_sents(input_filename, base_np_sentences, constraints, unfiltered_sentences):
    
    token_ctr = 0
    sent_ctr = 0
    unfilt_sent_ctr = 0
    reformatted_sents = []
    reformatted_sent = []
    with open(input_filename) as f:
        for line in f:
            line = line.strip()
            if line not in string.whitespace:
                line = line.split("\t")
                tokens = line[1].split(BASE_NP_SEPARATOR)
                reformatted_tokens = []
                reformatted_token_string = ""
                seen_basenp = False
                for token in tokens:
                    if base_np_sentences[sent_ctr][token_ctr] != token:
                        assert base_np_sentences[sent_ctr][token_ctr] == SONP_SYM
                        assert len(reformatted_tokens) == 0
                        reformatted_tokens.append("_")
                        assert base_np_sentences[sent_ctr][token_ctr+1] == token
                        #reformatted_tokens.append(token)
                        #print sent_ctr, unfilt_sent_ctr
                        reformatted_tokens.append(unfiltered_sentences[sent_ctr][unfilt_sent_ctr])
                        token_ctr += 1
                        seen_basenp = True
                    else:
                        #reformatted_tokens.append(token)
                        reformatted_tokens.append(unfiltered_sentences[sent_ctr][unfilt_sent_ctr])
                    unfilt_sent_ctr += 1
                    token_ctr += 1
                assert len(reformatted_tokens) >= 1
                if len(reformatted_tokens) > 1: # have seen a base NP and need to end EONP_SYM
                    assert base_np_sentences[sent_ctr][token_ctr] == EONP_SYM
                    assert seen_basenp
                    token_ctr += 1
                    reformatted_tokens.append("_")
                    reformatted_token_string = "_".join(reformatted_tokens)
                else:
                    reformatted_token_string = reformatted_tokens[0]
                #print reformatted_token_string
                pos_tag = line[3]
                head_id = str(int(line[6]) - ZGEN_OFFSET)
                deprel = line[7]
                reformatted_row = []
                reformatted_row.append(reformatted_token_string)
                if "__" in reformatted_token_string:
                    pos_tag = "NP"
                #if CONSTANT_DEP_LABEL:
                #    deprel = "N"
                if constraints == "Ref":
                    reformatted_row.extend([pos_tag, head_id, deprel])
                elif constraints == "OnlyPos":
                    reformatted_row.extend([pos_tag, "-1", "_"])
                elif constraints == "ArcsPos":
                    reformatted_row.extend([pos_tag, head_id, "_"])
                elif constraints == "OnlyArcs":
                    reformatted_row.extend(["-NONE-", head_id, "-NONE-"])
                elif constraints == "Input":
                    reformatted_row.extend(["-NONE-", "-1", "-NONE-"])
                reformatted_sent.append("\t".join(reformatted_row))
            else:
                token_ctr = 0
                unfilt_sent_ctr = 0
                sent_ctr += 1
                assert len(reformatted_sent) > 0
                reformatted_sents.append("\n".join(reformatted_sent) + "\n")
                reformatted_sents.append("\n")
                reformatted_sent = []
    
    # check that the final sentence was added
    assert reformatted_sent == []
    return reformatted_sents
    
def save_output(filename, list_of_strings, output_folder):
    
    with open(path.join(output_folder, filename), "w") as f:
        f.writelines(list_of_strings)

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ptb_dir', help="Directory containing wsj")
    parser.add_argument('--filtered_dep_dir', help="Directory containing filtered dependency trees")
    parser.add_argument('--zgen_output_dir', help="Directory for saving ZGen formatted data.")
    parser.add_argument('--zgen_output_dir_nonpsyms', help="Directory for saving ZGen formatted data (without BNPs).")
    
    args = parser.parse_args(arguments)
    
    ptb_dir = args.ptb_dir 
    filtered_dep_dir = args.filtered_dep_dir   
    zgen_output_dir_bnps = args.zgen_output_dir
    zgen_output_dir_nobnps = args.zgen_output_dir_nonpsyms
    train_words, valid_words, test_words, train_bnps, valid_bnps, test_bnps = get_bnp_from_ptb(ptb_dir)
    
    for zgen_output_dir, include_bnps in zip([zgen_output_dir_bnps, zgen_output_dir_nobnps], [True, False]):
        for split_name in ["train", "valid", "test"]:
            if split_name == "train":
                if include_bnps:
                    base_np_sentences = train_bnps
                else:
                    base_np_sentences = train_words
                unfiltered_sentences = train_words
            elif split_name == "valid":
                if include_bnps:
                    base_np_sentences = valid_bnps
                else:
                    base_np_sentences = valid_words
                unfiltered_sentences = valid_words
            elif split_name == "test":
                if include_bnps:
                    base_np_sentences = test_bnps
                else:
                    base_np_sentences = test_words
                unfiltered_sentences = test_words
            if include_bnps:
                intput_file = path.join(filtered_dep_dir, "{split_name}_filtered_dep_projected.txt".format(split_name=split_name))
            else:
                intput_file = path.join(filtered_dep_dir, "{split_name}_filtered_dep_nonpsyms_projected.txt".format(split_name=split_name))
                                    
            reformatted_sents = get_reformatted_sents(intput_file, base_np_sentences, "Ref", unfiltered_sentences)
            filename = "{split_name}_ref.txt".format(split_name=split_name)
            save_output(filename, reformatted_sents, zgen_output_dir)

            # create the POS dictionary for ZGen
            if split_name == "train":
                # tokens should be able to appear multiple times with different POS tags (e.g., "Does")
                # but mutliple word, POS pairs shouldn't be duplicated (e.g., "Does"\t"VBZ" appears once)
                posdict = defaultdict(list)
                for sent in reformatted_sents:
                    if sent.strip() not in string.whitespace:
                        #print sent
                        sent = sent.split("\n")
                        for row in sent:
                            if len(row) > 1:
                                row = row.split("\t")
                                if row[0] in posdict:
                                    # check if the POS tag has already been added
                                    existing_postags = posdict[row[0]]
                                    if row[1] not in existing_postags:
                                        posdict[row[0]].append(row[1])
                                else:
                                    posdict[row[0]].append(row[1])
                
                posdict_list = []       
                for token in posdict:
                    for pos in posdict[token]:
                        line = token + "\t" + pos + "\n"
                        posdict_list.append(line)
                    
                filename = "{split_name}_posdict.txt".format(split_name=split_name)
                save_output(filename, posdict_list, zgen_output_dir)        
                                
            reformatted_sents = get_reformatted_sents(intput_file, base_np_sentences, "Input", unfiltered_sentences)
            filename = "{split_name}_in.txt".format(split_name=split_name)
            save_output(filename, reformatted_sents, zgen_output_dir)    
        
            reformatted_sents = get_reformatted_sents(intput_file, base_np_sentences, "OnlyArcs", unfiltered_sentences)
            filename = "{split_name}_in_tree.txt".format(split_name=split_name)
            save_output(filename, reformatted_sents, zgen_output_dir)  
                
            #reformatted_sents = get_reformatted_sents(intput_file, base_np_sentences, "ArcsPos", unfiltered_sentences)
            #filename = "{split_name}_filtered_projected_dep_arc_and_pos_constraints.txt".format(split_name=split_name)
            #save_output(filename, reformatted_sents, zgen_output_dir)    
            #
            #reformatted_sents = get_reformatted_sents(intput_file, base_np_sentences, "OnlyArcs", unfiltered_sentences)
            #filename = "{split_name}_in_nopos.txt".format(split_name=split_name)
            #save_output(filename, reformatted_sents, zgen_output_dir)
            #
            #reformatted_sents = get_reformatted_sents(intput_file, base_np_sentences, "All", unfiltered_sentences)
            #filename = "{split_name}_in_nopos.txt".format(split_name=split_name)
            #save_output(filename, reformatted_sents, zgen_output_dir)
        

            # Also save the corresponding files for the Yara parser:
            if not include_bnps:
                reformatted_sents = get_reformatted_sents_yara(intput_file, base_np_sentences, "YaraDep", unfiltered_sentences)
                filename = "{split_name}_yara_depv3.txt".format(split_name=split_name)
                save_output(filename, reformatted_sents, zgen_output_dir)
            
                reformatted_sents = get_reformatted_sents_yara(intput_file, base_np_sentences, "YaraPos", unfiltered_sentences)
                filename = "{split_name}_yara_posv3.txt".format(split_name=split_name)
                save_output(filename, reformatted_sents, zgen_output_dir)  
    
                                    
            
            
            
    print "Complete"            


    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))