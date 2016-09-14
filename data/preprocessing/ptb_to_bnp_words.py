"""

This script takes PTB constituency tress and adds base NP symbols (BNPs).

This script is part of a pipeline designed to replicate Liu and Zhang 2015.

Following previous work, base noun phrases are defined as noun phrases without 
nested constituents.

This requires NLTK. It has beem most recently tested with 
nltk.__version__=='3.0.4'


"""



from nltk.corpus import ptb
import nltk
import glob
import numpy as np
from os import path

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
    
    return_flat_tree: If True, base NP's are returned as nested lists
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
                                    #tokens.extend( [SONP_SYM] + [str(x) for x in subtree.leaves()] + [EONP_SYM])
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
    

    
def count_basenps(base_np_delineated_tokens):
    """
    Return length/count stats
    
    """
    
    basenp_count = 0
    token_count = 0
    seen_sonp = -1
    np_lens = []
    in_sonp = False
    all_lens = []
    for token in base_np_delineated_tokens:
        assert type(token) == type("str"), "%s %s" % (token, base_np_delineated_tokens)
        if token == SONP_SYM:
            basenp_count += 1
            seen_sonp = 0
            in_sonp = True
        elif token == EONP_SYM:
            np_lens.append(seen_sonp)
            all_lens.append(seen_sonp)
            in_sonp = False
        else:
            #if token != SONP_SYM and token != EONP_SYM:
            token_count += 1
            seen_sonp += 1
            if not in_sonp:
                all_lens.append(1)
    return basenp_count, token_count, np_lens, all_lens
    
def remove_base_np_syms(base_np_delineated_tokens_list):
    """
    Remove the base NP symbols. No preprocessing/filtering is applied
        
    """
    
    all_sent_tokens = []
    for base_np_delineated_tokens in base_np_delineated_tokens_list:
        tokens = []
        for token in base_np_delineated_tokens:
            assert type(token) == type("str"), "%s %s" % (token, base_np_delineated_tokens)
            
            if token != SONP_SYM and token != EONP_SYM:
                tokens.append(token)
        all_sent_tokens.append(tokens)
    return all_sent_tokens


   
def get_bnp_from_ptb(ptb_dir):
    #train_fileids = [ptb_dir+"%02d"%x for x in range(2, 22)]
    #valid_fileids = [ptb_dir+"%02d"%x for x in range(22, 23)]
    #test_fileids = [ptb_dir+"%02d"%x for x in range(23, 24)]

    train_fileids = [path.join(ptb_dir, "%02d"%x) for x in range(2, 22)]
    valid_fileids = [path.join(ptb_dir, "%02d"%x) for x in range(22, 23)]
    test_fileids = [path.join(ptb_dir, "%02d"%x) for x in range(23, 24)]
        
    train = []
    valid = []
    test = []
    basenp_count_train = 0
    basenp_count_valid = 0
    basenp_count_test = 0
    token_count_train = 0
    token_count_valid = 0
    token_count_test = 0
    np_lens_train = []
    np_lens_valid = []
    np_lens_test = []
    
    all_lens_train = []
    
    print_every = 1000
    
    
    for split_fileids, split_label in zip([train_fileids, valid_fileids, test_fileids], ["train", "valid", "test"]):
        sent_ctr = 0
        for wsj_section_folderpath in split_fileids:
            all_mrg_files_in_split = glob.glob(wsj_section_folderpath+"/*.mrg")
            for mrg_file in all_mrg_files_in_split:
                parsed_sents = ptb.parsed_sents(mrg_file)
                parsed_sents_ctr = 0
                for parsed_tree in parsed_sents:
                    if sent_ctr % print_every == 0:
                        print "Currently processing %d in %s" % (sent_ctr, split_label)
                    sent_ctr += 1
                    base_np_delineated_tokens = traverse_tree(parsed_tree, True)
                    
                    parsed_sents_ctr += 1
                    if split_label == "train":
                        
                        basenp_count_split, token_count_split, np_lens_split, all_lens_split = count_basenps(base_np_delineated_tokens)
                        basenp_count_train += basenp_count_split
                        token_count_train += token_count_split
                        np_lens_train.extend(np_lens_split)
                        
                        all_lens_train.extend(all_lens_split)
                        
                        train.append(base_np_delineated_tokens)
                    elif split_label == "valid":
                        
                        basenp_count_split, token_count_split, np_lens_split, _ = count_basenps(base_np_delineated_tokens)
                        basenp_count_valid += basenp_count_split
                        token_count_valid += token_count_split
                        np_lens_valid.extend(np_lens_split)
                        
                        valid.append(base_np_delineated_tokens)
                    elif split_label == "test":
                        
                        basenp_count_split, token_count_split, np_lens_split, _ = count_basenps(base_np_delineated_tokens)
                        basenp_count_test += basenp_count_split
                        token_count_test += token_count_split
                        np_lens_test.extend(np_lens_split)
                        
                        test.append(base_np_delineated_tokens)

    print "Train"
    print "Total bag size: %d, Average size of item in bag %f" % (len(all_lens_train), np.mean(all_lens_train))
    print "Base NP count: %d, Average Base NP length: %f, Token count: %d, Sentence count: %d" % (basenp_count_train, np.mean(np_lens_train), token_count_train, len(train))
    
    print "Valid"
    print "Base NP count: %d, Average Base NP length: %f, Token count: %d, Sentence count: %d" % (basenp_count_valid, np.mean(np_lens_valid), token_count_valid, len(valid))
    
    print "Test"
    print "Base NP count: %d, Average Base NP length: %f, Token count: %d, Sentence count: %d" % (basenp_count_test, np.mean(np_lens_test), token_count_test, len(test))
    
    
    
    
    #startswith("NP"): Number of base NPs in the training set: 228399
    #Train
    #Base NP count: 228399, Average Base NP length: 2.212983, Token count: 949938, Sentence count: 39832
    #Valid
    #Base NP count: 9536, Average Base NP length: 2.273700, Token count: 40104, Sentence count: 1700
    #Test
    #Base NP count: 13457, Average Base NP length: 2.192465, Token count: 56674, Sentence count: 2416
    #Size of vocab (i.e., tokens that appear in training, not including the additional <unk>) 10000
    
    
    raw_train = remove_base_np_syms(train)
    raw_valid = remove_base_np_syms(valid)
    raw_test = remove_base_np_syms(test)
    
    
    return raw_train, raw_valid, raw_test, train, valid, test


