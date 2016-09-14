"""

This script is designed to align NP-bracketed PTB dependency trees to tokens with
base NPs derived from the constinuency trees.

The bulk of this code exists to handle edge cases introduced by
re-attaching heads when collapsing tokens within BNPs.

Some edge cases are under-specified in previous work. Our work is internally
valid in that we used the same BNPs across experiments (syntax and no-syntax),
and externally valid in that we verified results (using the NGram model) using the BNP data 
subsequently acquired of previous work.

"""

from collections import defaultdict
import string

from os import path



NUMERIC_SYM = "N"
LOW_COUNT_SYM = "<unk>"
EOS_SYM = "<eos>"
BASE_NP_SEPARATOR = "^^"
SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"

  
    
def write_list_of_list_of_list(list_of_list_of_of_list_of_strings, filename_with_path, add_newline = True, add_space = True, split_with_tabs = True):
    if split_with_tabs:
        split_string = "\t"
    else:
        split_string = " "
    
    with open(filename_with_path, "w") as f:
        
        for list_of_list_of_strings in list_of_list_of_of_list_of_strings:
            for list_of_strings in list_of_list_of_strings:
                if add_newline:
                    f.write(split_string.join(list_of_strings) + "\n")
                else:
                    f.write(split_string.join(list_of_strings))            
            if add_space:
                f.write("\n")  


       

def preprocess_token_with_tag_simple(collapsed_row_ctr, wsj_filtered_tokens_one_sent, token, tag, strip_low_freq=False, dependency=False):
    if dependency:
        if tag in ["-NONE-"]: 
            return None     
        
    if token == "(":
        token = "-LRB-"
    elif token == ")":
        token = "-RRB-"
    elif token == "{":
        token = "-LCB-"
    elif token == "}":
        token = "-RCB-"            
            
    if collapsed_row_ctr > (len(wsj_filtered_tokens_one_sent) - 1) or wsj_filtered_tokens_one_sent[collapsed_row_ctr] != token:
        return None
    return token
    
  

def collapse_npsyms(full_tree, tokens_one_sent_npsyms, ctr):
    token_to_head = {}
    for row in full_tree:
        token_id = row[0]
        head_id = row[6]
        assert token_id not in token_to_head
        token_to_head[token_id] = head_id    
        
    removed_ids = defaultdict(str)
    
    
    npsyms_ctr = 0
    collapsed_tree = []
    i = 0
    
    tokenid_to_basenpid = {}
    basenpid_to_tokenid = {}
    basenpid_to_identity_tokenid = {}
    basenpid = 0
    while (i < len(full_tree)): 
        row = full_tree[i]
    
        token_id = row[0]
        token = row[1]

        
        basenp_str = ""
        if tokens_one_sent_npsyms[npsyms_ctr] == SONP_SYM:
            npsyms_ctr += 1
            
            while True:
                assert tokens_one_sent_npsyms[npsyms_ctr] == full_tree[i][1]
                if len(basenp_str) > 0:
                    basenp_str += BASE_NP_SEPARATOR
                basenp_str += full_tree[i][1]
                
                
                if tokens_one_sent_npsyms[npsyms_ctr+1] == EONP_SYM:
                    original_row = list(full_tree[i])
                    # the only change is to the token string; the other fields of this rightmost token are retained as is
                    original_row[1] = basenp_str
                    tokenid_to_basenpid[full_tree[i][0]] = str(basenpid)
                    if str(basenpid) not in basenpid_to_tokenid:
                        basenpid_to_tokenid[str(basenpid)] = [full_tree[i][0]]
                    else:
                        basenpid_to_tokenid[str(basenpid)].append(full_tree[i][0])
                    basenpid_to_identity_tokenid[str(basenpid)] = original_row[0]
                    collapsed_tree.append(original_row)
                    npsyms_ctr += 2
                    i += 1
                    basenpid += 1
                    break
                else:
                    tokenid_to_basenpid[full_tree[i][0]] = str(basenpid)
                    if str(basenpid) not in basenpid_to_tokenid:
                        basenpid_to_tokenid[str(basenpid)] = [full_tree[i][0]]
                    else:
                        basenpid_to_tokenid[str(basenpid)].append(full_tree[i][0])
                    removed_ids[full_tree[i][0]] = 1 # current row id
                    npsyms_ctr += 1
                    i += 1
                
        else:
            # outside a base NP
            assert token == tokens_one_sent_npsyms[npsyms_ctr]
            collapsed_tree.append(list(row))
            npsyms_ctr += 1
            i += 1    

    
    assert "0" not in removed_ids
    
    token_to_head_filtered = {} # defaultdict(str)

    for row in collapsed_tree:
        token_id = row[0]
        head_id = row[6]
        next_head = head_id
        
        if token_id in tokenid_to_basenpid: # in a base NP 
            curr_basenp_id = tokenid_to_basenpid[token_id]
            while True:   
                if next_head in removed_ids or next_head == token_id:
                    if next_head in tokenid_to_basenpid and curr_basenp_id != tokenid_to_basenpid[next_head]:
                        basenpid = tokenid_to_basenpid[next_head]
                        next_head = basenpid_to_identity_tokenid[basenpid]
                        if next_head != token_id:
                            break
                    else:
                        next_head = token_to_head[next_head]
                    
                else:
                    if next_head != token_id:
                        break 
        else:
            while True:
                if next_head in removed_ids:
                    if next_head in tokenid_to_basenpid:
                        basenpid = tokenid_to_basenpid[next_head]
                        next_head = basenpid_to_identity_tokenid[basenpid]
                        if next_head != token_id:
                            break
                    else:
                        next_head = token_to_head[next_head]
                    
                    
                else:
                    if next_head != token_id:
                        break            
        # now, ensure there's not a loop back to the new head_id
        # if so, continue traversing:
        assert next_head not in removed_ids
        candidate_next_head = next_head
        if candidate_next_head in token_to_head_filtered and token_to_head_filtered[candidate_next_head] == token_id:
            # find nearest remaining head
            next_head = token_to_head[candidate_next_head]
            while next_head in removed_ids or next_head == token_id:
                next_head = token_to_head[next_head]
                         
        token_to_head_filtered[token_id] = next_head

                          
                 
    # make the reamining indecies consecutive
    original_id_to_new_id = defaultdict(str)

    # add root:
    token_to_head_filtered["0"] = "0" 
    original_id_to_new_id["0"] = "0"
            
    decrement = 0
    for i in range(1, len(full_tree)+1):
        str_i = str(i)
        if str_i not in token_to_head_filtered:
            decrement += 1
        else:
            if decrement == 0:
                original_id_to_new_id[str_i] = str_i
            else:
                original_id_to_new_id[str_i] = str(i - decrement)

    # finally, make ConLL format with updated ids and heads
    number_of_roots = 0
    renumbered_filtered_collapsed_tree = []
    for row in collapsed_tree:
        old_row = list(row)
        token_id = old_row[0]
        head_id = old_row[6]

        new_row = list(row)
        new_token_id = original_id_to_new_id[token_id]
        new_row[0] = new_token_id
        
        new_head_id = token_to_head_filtered[token_id]
        new_head_id_with_updated_numbering = original_id_to_new_id[new_head_id]
        new_row[6] = new_head_id_with_updated_numbering

        if new_head_id == "0":
            number_of_roots += 1
                
        renumbered_filtered_collapsed_tree.append(new_row)
        
    assert number_of_roots == 1, renumbered_filtered_collapsed_tree
    # final check of lengths
    num_nps_and_tokens = 0
    in_np = False
    for token in tokens_one_sent_npsyms:
        if in_np:
            if token == EONP_SYM:
                in_np = False
        else:
            if token == SONP_SYM:
                in_np = True
            num_nps_and_tokens += 1
    assert num_nps_and_tokens == len(renumbered_filtered_collapsed_tree)
    
    # ensure a proper tree is formed
    arcs = {}
    arcmap = defaultdict(str)
    for row in renumbered_filtered_collapsed_tree:
        token_id = row[0]
        head_id = row[6]    
        assert token_id not in arcmap
        arcmap[token_id] = head_id
        if head_id in arcs:
            arcs[head_id].append(token_id)
        else:
            arcs[head_id] = [token_id]
    
    assert "0" in arcs
    for row in renumbered_filtered_collapsed_tree:
        token_id = row[0]
        head_id = row[6]    
        assert head_id == "0" or head_id in arcmap
        head_id_head_id = arcmap[head_id]
        if head_id_head_id == token_id:
            print collapsed_tree
            print renumbered_filtered_collapsed_tree
        assert head_id_head_id != token_id, token_id 

    return renumbered_filtered_collapsed_tree

def get_descendants(target_id, arcmap):
    descendants = {}
    for phrase_id, head_id in arcmap.items():
        if head_id == target_id:
            descendants[phrase_id] = 1
    
            new_descendants = get_descendants(phrase_id, arcmap)
            for p, _ in new_descendants.items():
                descendants[p] = 1

    return descendants   
            
issues1 = 0 
issues2 = 0   
issues3 = 0  
forward_slash_found = 0 

def filter_tree(full_tree, wsj_filtered_tokens_one_sent, ctr, wsj_filtered_tokens_one_sent_npsyms):
    global issues1
    global issues2
    global issues3
    global forward_slash_found
    lines_with_issues = 0 
    # this could be more efficient, but for the purposes here (i.e., it's run once), multiple passes
    # over the tree make it easier to follow
    
    token_to_head = {}
    for row in full_tree:
        token_id = row[0]
        head_id = row[6]
        assert token_id not in token_to_head
        token_to_head[token_id] = head_id
        
    filtered_tokens = []
    
    removed_ids = defaultdict(str)
    
    #buy\/hold
    #indecies_to_collapse = []
    collapsed_tree = []
    i = 0
    tree_offset = 0
    while (i < len(full_tree)): 
        row = full_tree[i]
    
        token_id = row[0]
        token = row[1]
        tag = row[3]
        
        # The dependency conversion/Vadas' NP bracketing splits tokens with internal '/'
        # We need to regroup these and delete the extra arcs to match the gold LM sets
        # (this is a throwback to when we were matching the neural LM datasets, but 
        # also necessary to match the PTB tokenization)
        # Note that here we're following the overwhelming majority of cases 
        # in assigning the head of the group to be that of the *left-most* token
        if token == "/" and tag == "CC": # and i != (len(full_tree)-1):
            forward_slash_found += 1
            
            prev_row_token = collapsed_tree[i-1-tree_offset][1]

            next_row_token = full_tree[i+1][1]
            
            # here, we're removing the current row and the next row
            next_row_id = full_tree[i+1][0]
            removed_ids[next_row_id] = 1 # next row id
            removed_ids[token_id] = 1 # current row id
            
            prev_row_head_id = collapsed_tree[i-1-tree_offset][6]
            prev_row_token_id = collapsed_tree[i-1-tree_offset][0]
            cur_row_head_id = row[6]
            cur_row_token_id = token_id
            next_row_head_id = full_tree[i+1][6]
            next_row_token_id = full_tree[i+1][0]
            
            if cur_row_head_id != prev_row_token_id:
                #print "failed cur_row_head_id == prev_row_token_id"
                issues3 += 1
            if next_row_head_id != cur_row_token_id:
                #print "failed next_row_head_id == cur_row_token_id"
                issues1 += 1
            if not ( (prev_row_head_id != cur_row_token_id) and (prev_row_head_id != next_row_token_id) ):
                #print "failed (prev_row_head_id != cur_row_token_id) and (prev_row_head_id != next_row_token_id)"
                issues2 += 1
            #print "-----------------"

            collapsed_tree[i-1-tree_offset][1] = prev_row_token + "\\/" + next_row_token
            tree_offset += 2
            i += 2
        else:
            collapsed_tree.append(list(row))
            
            i += 1

    filtered_collapsed_tree = [] 
       
    collapsed_row_ctr = 0
    for row in collapsed_tree:
        token_id = row[0].strip()
        token = row[1].strip()
        tag = row[3].strip()
        
        filtered_token = preprocess_token_with_tag_simple(collapsed_row_ctr, wsj_filtered_tokens_one_sent, token, tag, strip_low_freq=False, dependency=True)

        if filtered_token:
            filtered_tokens.append(filtered_token)
            filtered_row = list(row)
            filtered_row[1] = filtered_token
            filtered_collapsed_tree.append(filtered_row)
            collapsed_row_ctr += 1 # there's an assumption that the ellision here is only in the base-NP data
        else:
            removed_ids[token_id] = 1
        
            
            
    # two final passes to make the indecies consecutive (since some may have been dropped above),
    # and to reattach orphaned arcs
    # The latter can occur as in the following, where the parentheses were dropped
        """
        14	dial	_	VB	_	_	13	IM	_	_
        15	(	_	(	_	_	14	PRN	_	_
        16	and	_	CC	_	_	15	DEP	_	_
        17	redial	_	VB	_	_	15	DEP	_	_
        18	)	_	)	_	_	15	P	_	_
        19	movie	_	NN	_	_	22
        """
    # first, reattach heads:
    # recall, there are 2 ways a head can be dropped: 1. removing '/' split and 2. filtering
    #head_to_targets
    #token_to_head
    
    assert "0" not in removed_ids
    
    token_to_head_filtered = defaultdict(str)

    for row in filtered_collapsed_tree:
        token_id = row[0]
        head_id = row[6]
        #assert head_id not in removed_ids, "Removed id's shouldn't be heads"
        if head_id in removed_ids:
            # find nearest remaining head
            next_head = token_to_head[head_id]
            while next_head in removed_ids:
                next_head = token_to_head[next_head]
              
            token_to_head_filtered[token_id] = next_head
        else:
            token_to_head_filtered[token_id] = head_id
      
    
                 
    # make the remaining indecies consecutive
    original_id_to_new_id = defaultdict(str)

    # add root:
    token_to_head_filtered["0"] = "0" 
    original_id_to_new_id["0"] = "0"
            
    decrement = 0
    for i in range(1, len(full_tree)+1):
        str_i = str(i)
        if str_i not in token_to_head_filtered:
            decrement += 1
        else:
            if decrement == 0:
                original_id_to_new_id[str_i] = str_i
            else:
                original_id_to_new_id[str_i] = str(i - decrement)

    # finally, make ConLL format with updated ids and heads
    renumbered_filtered_collapsed_tree = []
    for row in filtered_collapsed_tree:
        old_row = list(row)
        token_id = old_row[0]
        head_id = old_row[6]
        
        new_row = list(row)
        new_token_id = original_id_to_new_id[token_id]
        new_row[0] = new_token_id
        
        new_head_id = token_to_head_filtered[token_id]
        new_head_id_with_updated_numbering = original_id_to_new_id[new_head_id]
        new_row[6] = new_head_id_with_updated_numbering
        
        renumbered_filtered_collapsed_tree.append(new_row)
        
    
    #if ctr in [26, 658, 763]:        
    if filtered_tokens != wsj_filtered_tokens_one_sent:
        
        print filtered_tokens
        print wsj_filtered_tokens_one_sent
        lines_with_issues += 1
        #print lines_with_issues
        import pdb; pdb.set_trace()
    
            
    renumbered_filtered_collapsed_tree = collapse_npsyms(list(renumbered_filtered_collapsed_tree), list(wsj_filtered_tokens_one_sent_npsyms), ctr)        
    return renumbered_filtered_collapsed_tree, lines_with_issues
        
def check_tree_yields(filter_tree_result, wsj_filtered_tokens_one_sent):
    tree_yield = []
    for row in filter_tree_result:
        tree_yield.append(row[1])
    assert tree_yield == wsj_filtered_tokens_one_sent
    
    
def get_filtered_dependency_trees(dependency_output_dir, split_label, wsj_filtered_tokens, wsj_filtered_tokens_npsyms):
    
    if split_label == "train":
        split_fileids = ["wsj_%02d_dep.txt"%x for x in range(2, 22)]
    elif split_label == "valid":
        split_fileids = ["wsj_%02d_dep.txt"%x for x in range(22, 23)]
    else:
        split_fileids = ["wsj_%02d_dep.txt"%x for x in range(23, 24)]
    
    full_trees = []
    full_trees_filtered = []
    
    total_lines_with_possible_issues = 0
    ctr = 0
    for split_fileid in split_fileids:
        print "split_fileid:", split_fileid
        with open(path.join(dependency_output_dir, split_fileid)) as f:
            full_tree = []
            for line in f:
                if line not in string.whitespace:
                    full_tree.append(line.split())   
                    assert len(full_tree[-1]) == 10                     
                else:
                    if len(full_tree) != 0:
                        full_trees.append(full_tree)
                        filter_tree_result, lines_with_issues = filter_tree(full_tree, wsj_filtered_tokens[ctr], ctr, wsj_filtered_tokens_npsyms[ctr])
                        #check_tree_yields(filter_tree_result, wsj_filtered_tokens[ctr])
                        total_lines_with_possible_issues += lines_with_issues
                        full_trees_filtered.append(filter_tree_result)
                        full_tree = []
                        ctr += 1
            # fencepost (in case trailing new line is missing)
            if len(full_tree) != 0:
                full_trees.append(full_tree)
                filter_tree_result, lines_with_issues = filter_tree(full_tree, wsj_filtered_tokens[ctr], ctr, wsj_filtered_tokens_npsyms[ctr])
                #check_tree_yields(filter_tree_result, wsj_filtered_tokens[ctr])
                total_lines_with_possible_issues += lines_with_issues
                full_trees_filtered.append(filter_tree_result)
                full_tree = []   
                ctr += 1             
    print "Total lines with possible issues:", total_lines_with_possible_issues   
    return full_trees, full_trees_filtered                        
                                                                 


def save_dependency_trees(train_words, valid_words, test_words, train_bnps, valid_bnps, test_bnps, dependency_dir, filtered_dependency_dir, include_bnps):
    for words, bnps, split_label in zip([train_words, valid_words, test_words], [train_bnps, valid_bnps, test_bnps], ["train", "valid", "test"]):
        if include_bnps:
            full_trees, filtered_trees = get_filtered_dependency_trees(dependency_dir, split_label, words, bnps)   
            filename_suffix = "_filtered_dep.txt"
        else:
            full_trees, filtered_trees = get_filtered_dependency_trees(dependency_dir, split_label, words, words)   
            filename_suffix = "_filtered_dep_nonpsyms.txt"
        write_list_of_list_of_list(filtered_trees, path.join(filtered_dependency_dir, "%s%s" % (split_label, filename_suffix)), add_newline = True, add_space = True, split_with_tabs = True)
        
    