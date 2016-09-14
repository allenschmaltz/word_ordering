"""
version 2

This scripts takes input files in the ZGen format and converts them to sentences (one per
line, with EOS symbols) for use with training and testing the LSTM models.
  
Here, case is retained. <unk> is used for low-frequency tokens appearing in
training, as well as tokens in the validation and testing splits which never
occur in the training split. Optionally, a unique low-frequency token is used
for upper-case types.

(Functions for generating word-action sequences based on a modification of
the arc-standard format where actions are delayed until seeing the top of the
stack are also included in this file, for historical reference.
We refer to this as 'arc lazy' format.
Used as a prefix parser, we found that delaying the actions improved performance.
However, these are not used in the current word-ordering paper results and
should be considered alpha.) 

"""


import string

import sys
import argparse
from os import path
import random
from collections import defaultdict, deque

random.seed(1776)

SONP_SYM = "<sonp>"
EONP_SYM = "<eonp>"
EOS_SYM = "<eos>"
NUMERIC_SYM = "N"
LOW_COUNT_SYM = "<unk>"
LOW_COUNT_SYM_UPPER = "<Unk>"
ARC_LABEL = "_"

WORD_ACTION_SEPARATOR = "@"
ROOT_SYM = WORD_ACTION_SEPARATOR + "T"
INCLUDE_ROOT = False
NULL_SYM = WORD_ACTION_SEPARATOR + "Q"
L_REDUCE_SYM = WORD_ACTION_SEPARATOR + "L"
R_REDUCE_SYM = WORD_ACTION_SEPARATOR + "R"
STACK_SEPARATOR = "<eos>"

RETAIN_UNK_CASE = False

#NUM_WORD_TYPES_RETAINED = 9999
#WORD_FREQ_CUTOFF = 3 # types that appear with frequency less than this constant in the training data will be replaced with the LOW_COUNT_SYM

def add_np_symbols_to_word_actions(sentence_word_actions, strip_low_freq, types_retained):
    sentences = []
    sentence = []
    sentences_arc_lazy = []
    #sentence_arc_lazy = []
    for sentence_word_action in sentence_word_actions:
        for word_action in sentence_word_action:
            tokens = word_action.split("__") # base NPs are marked with starting and trailing "__"
            if len(tokens) == 3: # base NP
                assert tokens[0] == "" and tokens[2] == "", "ERROR: The following base NP is malformed: %s" % word_action
                # base NP constituent members are separated with "_"
                preprocessed_tokens = []
                for one_token in tokens[1].split("_"):
                    preprocessed_tokens.append(preprocess_token(one_token, strip_low_freq, types_retained))
                # add deterministic arcs to the base NP
                base_np = [SONP_SYM] + preprocessed_tokens + [EONP_SYM] + [L_REDUCE_SYM] * (len(preprocessed_tokens)-1)
                sentence.extend(base_np)
                
            elif len(tokens) == 1: # not a base NP
                if (tokens[0].startswith(L_REDUCE_SYM)) or (tokens[0].startswith(R_REDUCE_SYM)) or (tokens[0] == EOS_SYM):
                    # the non-word symbols are not in types_retained
                    preprocessed_token = tokens[0]
                else:
                    preprocessed_token = preprocess_token(tokens[0], strip_low_freq, types_retained)
                sentence.append(preprocessed_token)
        sentences.append(" ".join(sentence) + "\n")
        sentences_arc_lazy.append(" ".join(get_arc_lazy_sentence(sentence)) + "\n")
        sentence = []
    return sentences, sentences_arc_lazy
    
def get_arc_lazy_sentence(sentence):
    arc_lazy_line = []
    current_action_seq = []
    seen_action = False
    delayed_end = False
    for i, word_action in enumerate(sentence):
        if (word_action.startswith(L_REDUCE_SYM)) or (word_action.startswith(R_REDUCE_SYM)):
            seen_action = True
            current_action_seq.append(word_action)
        else:
            if seen_action and word_action not in [SONP_SYM, EONP_SYM]:
                if len(sentence) > i and sentence[i+1] == EONP_SYM:
                    delayed_end = True
                    arc_lazy_line.append(word_action)
                else:
                    arc_lazy_line.append(word_action)
                    arc_lazy_line.extend(current_action_seq)
                    current_action_seq = []
                    seen_action = False
            else:
                if delayed_end: # put trailing actions after the EONP_SYM
                    assert word_action == EONP_SYM
                    delayed_end = False
                    arc_lazy_line.append(word_action)
                    arc_lazy_line.extend(current_action_seq)
                    current_action_seq = []
                    seen_action = False 
                else:               
                    arc_lazy_line.append(word_action)
    arc_lazy_line.append(NULL_SYM)
    arc_lazy_line.append(R_REDUCE_SYM)
    return arc_lazy_line
    
def preprocess_token(token, strip_low_freq, types_retained):
    """
    
    Note that the global constant RETAIN_UNK_CASE determines whether case is retained
        within low frequency tokens.
        
    """
    
    # as an initial rough transformation, convert to N any token with a digit (may include parts of prices, dates, numbers with comma separators, fractions, etc.)
    for c in token:
        if c.isdigit():
            token = NUMERIC_SYM
            break
    # for the time being, retain punctuation 
    if strip_low_freq and types_retained:
        if token not in types_retained:
            if RETAIN_UNK_CASE:
                if token[0].isupper():
                    token = LOW_COUNT_SYM_UPPER
                else:
                    token = LOW_COUNT_SYM
            else:
                token = LOW_COUNT_SYM
    return token
    
def get_words_and_arcs(file_with_path, types_retained, arc_label, preprocess_tokens):                  
    """
    
    this version of get_words_and_arcs() handles the ZGen format
    """

    sentence_arcs = [] # [[[head index, arc label, target index]_0, ... , []_n ]_one_sent,...], where n is the number of tokens in the sentence
    sentence_id2w = [] # [{token index: token string,...}_one_sent,...]
    
    arcs = [] # arcs for one sentence
    id2w = {} # word to index dictionary for one sentence
    
    ctr = 0
    w_id_ctr = 1
    with open(file_with_path) as f:
        for line in f:
            
            if line not in string.whitespace:
                columns = line.split()
                w = columns[0]
                head_id = columns[2]
        
                assert head_id.isdigit() or head_id == "-1", "The head id is malformed in line %s" % line
                assert w != "", "The word token in line %s is unexpectedly missing" % line
                
                w_id = w_id_ctr
                head_id = int(head_id) + 1 # the 1 accounts for the ZGen offset
                assert head_id >= 0
                assert w_id != head_id, "%s %d %d" % (line, w_id, head_id)
                
                arc = [head_id, arc_label, w_id]
                arcs.append(arc)
                
                assert w_id not in id2w
                
                if preprocess_tokens:
                    w = preprocess_token(w, True, types_retained)
                
                id2w[w_id] = w
                w_id_ctr += 1
            else:
                sentence_arcs.append(arcs)
                arcs = []
                sentence_id2w.append(id2w)
                id2w = {}
                w_id_ctr = 1
                if ctr % 500 == 0:
                    print "Finished processing sentence %d" % ctr
                ctr+=1
    
    if len(arcs) != 0: # final sentence not yet added
        sentence_arcs.append(arcs)
        arcs = []
        sentence_id2w.append(id2w)
        id2w = {}                

    return sentence_arcs, sentence_id2w
    

def generate_gold_word_actions_stack(sentence_arcs, sentence_id2w, arc_label):
    
    """
    Generate the oracle word actions using the convention that reduce operations
    are applied to the top two tokens on the stack.
    
    """
    
    sentence_word_actions = [] # root | tokens | actions
    sentence_words = [] # the words in order (without appended actions)
    
    for i in xrange(0, len(sentence_id2w)):
    #if True:
    #    i = 0    
        id2w = sentence_id2w[i]
        #id2w[0] = "@T"
        arcs = sentence_arcs[i]
        word_ids = sorted(id2w.keys())
        
        # initialize c with ROOT and the first word of the sentence
        stack = [0]
        stack.append(word_ids[0])
        queue = deque(word_ids[1:])
        #c = [stack, queue, list(arcs)]
        used_arcs = []
        configurations = []
        
        observed_word_ids = {}
        word_actions = []
        
        word_actions.append("@T")
        
        #while len(queue) != 0:
        #while len(stack) > 1:
        while (len(queue) != 0 and len(stack) > 0) or (len(queue) == 0 and len(stack) > 1):
            # add word and separator to word-action sequence, if the word is seen for first time (in left-to-right order)
            # Note that the final actions on Root (RS: right arc and a shift) will be associated with the 
            # final word, which is typically punctuation in the raw data (but non-punctuation in the pre-processed data).
            if (len(stack) > 1) and stack[-1] not in observed_word_ids:
                observed_word_ids[stack[-1]] = True
                word_string = id2w[stack[-1]]
                #word_actions.append(word_string+WORD_ACTION_SEPARATOR)        
                word_actions.append(word_string)
            # choose the oracle transition t <- o(c)
            # case 1: left-arc
            if (len(stack) > 1) and [stack[-1], arc_label, stack[-2]] in arcs:
                used_arcs.append([stack[-1], arc_label, stack[-2]])
                configurations.append("L")
                stack0 = stack.pop()
                stack.pop()
                stack.append(stack0)
            # case 2: right-arc
            elif (len(stack) > 1) and [stack[-2], arc_label, stack[-1]] in arcs:
                # check that children have been added:
                children_added = True
                for word_id in word_ids:
                    if ([stack[-1], arc_label, word_id] in arcs) and not ([stack[-1], arc_label, word_id] in used_arcs):
                        children_added = False
                if children_added:
                    used_arcs.append([stack[-2], arc_label, stack[-1]])
                    configurations.append("R")
                    stack.pop() # get rid of stack0
                else:
                    configurations.append("S")
                    queue0 = queue.popleft()
                    stack.append(queue0)
            else:
                configurations.append("S")
                queue0 = queue.popleft()
                stack.append(queue0)
                
            # add action to word-action sequence
            action = configurations[-1]
            #word_action_sequence = word_actions[-1]
            # update:
            #word_actions[-1] = word_action_sequence + action
            if action != "S":
                word_actions.append(WORD_ACTION_SEPARATOR + action)
            
        ## START check that all words have been covered    
        words_check = []
        for sym in word_actions:
            if sym not in ["@T", "@L", "@R"]:
                words_check.append(sym)
        assert len(words_check) == len(id2w) == len(arcs)
        ## END check
        
        sentence_word_actions.append(word_actions)
        words_with_eos = [id2w[idx] for idx in word_ids] # ROOT is not included
        words_with_eos.append(STACK_SEPARATOR) 
        sentence_words.append(words_with_eos) 
    assert len(queue) == 0, "The queue/buffer is not 0 after generating the oracle. This suggests a problem, possibly with the ROOT arc"
    
    if INCLUDE_ROOT:
        assert False, "The current convention requires removing ROOT"
        return sentence_word_actions, sentence_words
    else:
         # strip the root and final @R   
        filtered_sentence_word_actions = []
        for word_actions in sentence_word_actions:
            assert word_actions[0] == ROOT_SYM
            assert word_actions[-1] == "@R"
            word_actions_filtered = list(word_actions[1:-1])
            # add <eos> and the final reduce-right action:
            word_actions_filtered.append(STACK_SEPARATOR)
            word_actions_filtered.append("@R")
            filtered_sentence_word_actions.append(word_actions_filtered)
        return filtered_sentence_word_actions, sentence_words
         
def get_word_types_retained(sentences_with_np_symbols, word_freq_cutoff):    
    word_to_freq = defaultdict(int)
    for sentence in sentences_with_np_symbols:
        sentence = sentence.split()
        for token in sentence:
            if token not in [SONP_SYM, EONP_SYM, EOS_SYM]: # NUMERIC_SYM will be included
                word_to_freq[token] += 1
    
    print "Total number of types:", len(word_to_freq)            

    word_freq_sorted = sorted(word_to_freq.items(), key=lambda x: x[1], reverse=True)
    #print word_freq_sorted
    #exit()
    
    types_retained = {}
    for word_type, word_freq in word_freq_sorted: #[0:NUM_WORD_TYPES_RETAINED]:
        if word_freq >= word_freq_cutoff:
            types_retained[word_type] = word_freq
    
    print "Total number of types after filtering:", len(types_retained)
    #print types_retained
    #exit()
    return types_retained
    
          
def _flatten_list(list_of_lists_and_strings):
    list_of_lists_and_strings = list(list_of_lists_and_strings)
    flattened_list = []
    for one_item in list_of_lists_and_strings:
        if type(one_item) != type([]):
            flattened_list.append(one_item)
        else:
            flattened_list.extend(one_item)
    return flattened_list

def _update_sentence_structures(sentences, sentences_shuffled, sentence, sentence_shuffled, add_eos):

    random.shuffle(sentence_shuffled)
    sentence_shuffled = _flatten_list(sentence_shuffled)
    if add_eos:
        sentence.append(EOS_SYM)
        sentence_shuffled.append(EOS_SYM)
    sentences.append(" ".join(sentence) + "\n")
    sentences_shuffled.append(" ".join(sentence_shuffled) + "\n")
    sentence = []
    sentence_shuffled = []   
    return sentences, sentences_shuffled, sentence, sentence_shuffled
      
def get_sentences_with_np_symbols(filename, add_eos, strip_low_freq, types_retained):
    sentences = []
    sentences_shuffled = []
    sentence = []
    sentence_shuffled = []
    
    with open(filename) as f:
        for line in f:
            if line not in string.whitespace:
                line = line.split("\t")
                tokens = line[0].split("__") # base NPs are marked with starting and trailing "__"
                if len(tokens) == 3: # base NP
                    assert tokens[0] == "" and tokens[2] == "", "ERROR: The following base NP is malformed: %s" % line[0]
                    # base NP constituent members are separated with "_"
                    preprocessed_tokens = []
                    for one_token in tokens[1].split("_"):
                        preprocessed_tokens.append(preprocess_token(one_token, strip_low_freq, types_retained))
                    
                    base_np = [SONP_SYM] + preprocessed_tokens + [EONP_SYM]
                    sentence.extend(base_np)
                    sentence_shuffled.append(list(base_np))
                elif len(tokens) == 1: # not a base NP
                    preprocessed_token = preprocess_token(tokens[0], strip_low_freq, types_retained)
                    sentence.append(preprocessed_token)
                    sentence_shuffled.append(preprocessed_token)
                else:
                    assert False, "ERROR: The following token string is malformed: %s" % line[0]
            else:
                # blank line indicates the end of the sentence:
                sentences, sentences_shuffled, sentence, sentence_shuffled = _update_sentence_structures(sentences, sentences_shuffled, sentence, sentence_shuffled, add_eos)

    if sentence != []:
        # in case the final sentence is missing a trailing blank line:
        sentences, sentences_shuffled, sentence, sentence_shuffled = _update_sentence_structures(sentences, sentences_shuffled, sentence, sentence_shuffled, add_eos)
    return sentences, sentences_shuffled

def save_list_of_lists(output_filename, list_of_lists):        
    with open(output_filename, "w") as f:
        f.writelines(list_of_lists) 
        
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-l', '--train_file', help="Training file in ZGen format.")
    parser.add_argument('-v', '--valid_file', help="Validation file in ZGen format.")
    parser.add_argument('-t', '--test_file', help="Testing file in ZGen format.")
    parser.add_argument('-o', '--output_dir', help="Output directory")
    parser.add_argument('-f', '--word_freq_cutoff', type=int, help="Words of freq less than this value in training are replaced with <unk>.", default=3)
    parser.add_argument('-r', '--retain_unk_case', help="Capitalized low freq words are replaced with <Unk>, whereas other low freq words are replaced with <unk>.", action="store_true")
    parser.add_argument('--lowercase_unk_sym', help="Defaults to <unk>.", default="<unk>")
    parser.add_argument('--uppercase_unk_sym', help="Used only with --retain_unk_case option. Defaults to <Unk>.", default="<Unk>")
    
    args = parser.parse_args(arguments)
       
    train_file = args.train_file
    valid_file = args.valid_file
    test_file = args.test_file
    output_dir = args.output_dir    
    word_freq_cutoff = args.word_freq_cutoff
    
    global RETAIN_UNK_CASE, LOW_COUNT_SYM, LOW_COUNT_SYM_UPPER
    RETAIN_UNK_CASE = args.retain_unk_case
    LOW_COUNT_SYM = args.lowercase_unk_sym
    LOW_COUNT_SYM_UPPER = args.uppercase_unk_sym
    
    word_types_retained = {}
    for split_name, split_file in zip(["train", "valid", "test"], [train_file, valid_file, test_file]):
        if split_name == "train":
            sentences_with_np_symbols, _ = get_sentences_with_np_symbols(split_file, True, False, None)
            word_types_retained = get_word_types_retained(sentences_with_np_symbols, word_freq_cutoff)
            sentences_with_np_symbols, sentences_with_np_symbols_shuffled = get_sentences_with_np_symbols(split_file, True, True, word_types_retained)
        else:
            sentences_with_np_symbols, sentences_with_np_symbols_shuffled = get_sentences_with_np_symbols(split_file, True, True, word_types_retained)
        
        output_filename = path.join(output_dir, "{SPLIT_NAME}_words_with_np_symbols.txt".format(SPLIT_NAME=split_name))
        save_list_of_lists(output_filename, sentences_with_np_symbols)
        output_filename = path.join(output_dir, "{SPLIT_NAME}_words_with_np_symbols_shuffled.txt".format(SPLIT_NAME=split_name))
        save_list_of_lists(output_filename, sentences_with_np_symbols_shuffled)

        ## generate word-actions:
        #sentence_arcs, sentence_id2w = get_words_and_arcs(split_file, None, ARC_LABEL, False)
        #sentence_word_actions, _ = generate_gold_word_actions_stack(sentence_arcs, sentence_id2w, ARC_LABEL) 
        #
        #sentences, sentences_arc_lazy = add_np_symbols_to_word_actions(sentence_word_actions, True, word_types_retained)
        #output_filename = path.join(output_dir, "{SPLIT_NAME}_arc_std_with_np_symbols.txt".format(SPLIT_NAME=split_name))
        #save_list_of_lists(output_filename, sentences)
        #output_filename = path.join(output_dir, "{SPLIT_NAME}_arc_lazy_with_np_symbols.txt".format(SPLIT_NAME=split_name))
        #save_list_of_lists(output_filename, sentences_arc_lazy)                    
            
            
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    


