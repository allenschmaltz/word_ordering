--[[ This is based in part on the code at https://github.com/wojzaremba/lstm/blob/master/data.lua,
    which containined the following license:
    
    ----  Copyright (c) 2014, Facebook, Inc.
    ----  All rights reserved.
    ----
    ----  This source code is licensed under the Apache 2 license found in the
    ----  LICENSE file in the root directory of this source tree. 
    ----

    The original LICENSE file is available at https://github.com/wojzaremba/lstm/blob/master/LICENSE
    A copy is available in license/Apache_LICENSE.txt.
---]]


local stringx = require('pl.stringx')
local file = require('pl.file')

local vocab_idx = 0
local vocab_map = {}


local function init_vocab_with_action_symbols(action_symbols)
  --[[
  Initializes the vocabulary mapping with the action symbols, if any.
  The action symbols are assigned ids from 1 to the number of action symbols.
  Returns the number of action symbols.
  --]]
  
  action_symbols_list = stringx.split(action_symbols,",")
  print("The following action symbols are being used:")
  print(action_symbols_list)
  
  for i = 1, #action_symbols_list do
    if vocab_map[action_symbols_list[i]] == nil then
       vocab_idx = vocab_idx + 1
       vocab_map[action_symbols_list[i]] = vocab_idx
    end
  end
   
  return #action_symbols_list
end

local function init_vocab_with_existing_vocab_map_and_existing_vocab_idx(existing_vocab_map, existing_vocab_idx)
  vocab_map = existing_vocab_map
  vocab_idx = existing_vocab_idx
end

local function get_vocab_map_and_vocab_idx()
  return vocab_map, vocab_idx
end


-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size (x_inp:size(1)/batch_size) x batch_size.
-- Note that some data may be cut in order to produce a rectangular output matrix
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end

-- The Stack is reset upon seeing <eos>, so it is important to not arbitrarily cut an internal <eos> symbol.
-- Instead, the tail of the data is truncated, as necessary, to produce a rectangular output matrix for batch learning.
local function replicate_cut_end(x_inp, batch_size)
   local s = x_inp:size(1)
   if torch.floor(s / batch_size) ~= (s / batch_size) then
     s = torch.floor(s / batch_size) * batch_size
     x_inp = x_inp[{{1,s}}]
   end
   
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end


local function load_data(perform_text_preprocessing_int, fname, new_vocab_warning)
  
   local data = file.read(fname)
   if perform_text_preprocessing_int == 0 then
     print("The data is being loaded with the expectation that applicable symbols have already been added. Newline '\\n' characters will be removed.")
     -- split() below removes newlines
   elseif perform_text_preprocessing_int == 1 then
      print("Adding <eos> symbols to the end of each line.")
      data = stringx.replace(data, '\n', ' <eos> ')
   else
     print("An invalid command line option was specified for -perform_text_preprocessing_int. Exiting")
     os.exit()
   end
   
   assert(stringx.lfind(data, "<eos>", 1) ~= nil, "The current convention expects <eos> symbols to be added. Consider using \z
     -perform_text_preprocessing_int 1.")
   
   local current_vocab_idx = vocab_idx
   
   data = stringx.split(data)
   
   print(string.format("Loading %s, size of data = %d", fname, #data))
   local x = torch.zeros(#data)
   for i = 1, #data do
      if vocab_map[data[i]] == nil then
         vocab_idx = vocab_idx + 1
         vocab_map[data[i]] = vocab_idx
         
      end
      x[i] = vocab_map[data[i]]
   end
   
   if new_vocab_warning then
     local added_vocab_idx = vocab_idx - current_vocab_idx
     if added_vocab_idx > 0 then
       print("WARNING: " .. added_vocab_idx .. " new tokens were added to the vocab_map.")
     end
   end
   
   return x
end

local function traindataset(batch_size, perform_text_preprocessing_int, input_path_with_filename)
   print("loading training data")
   local x = load_data(perform_text_preprocessing_int, input_path_with_filename, false)
   print("training data loaded")
   x = replicate_cut_end(x, batch_size)
   print("training data replicated")
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size, perform_text_preprocessing_int, input_path_with_filename)
   print("loading test data")
   if vocab_idx == 0 then
     print("ERROR: The vocabulary has not been initialized. Exiting.")
     os.exit()
   end
   
   local x = load_data(perform_text_preprocessing_int, input_path_with_filename, true)
   x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
   print("test data loaded")
   return x
end


local function validdataset(batch_size, perform_text_preprocessing_int, input_path_with_filename)
   print("loading validation data")
   if vocab_idx == 0 then
     print("ERROR: The vocabulary has not been initialized. Exiting.")
     os.exit()
   end
   
   local x = load_data(perform_text_preprocessing_int, input_path_with_filename, true)
   print("validation data loaded")
   x = replicate_cut_end(x, batch_size)
   print("validation data replicated")
   return x
end


local function load_unigram_lm(unigram_lm_path_with_filename, final_vocab_map, action_symbols_list, bnp, sonp_id, eonp_id)
  --[[
    Load an ARPA file containing unigram probabiliites, returning
    a map from token_id to log probability
    
    Assumes the ARPA file only contains unigram probabilities
    
    Assumes UNK for uppercase low-freq words, and unk for all other low-freq words
  --]]
  
  print("Loading the following ARPA file: " .. unigram_lm_path_with_filename)
  local token_id_to_unigram = {}

  local file = assert(io.open(unigram_lm_path_with_filename, "r"))

  local start_of_unigrams = false

  for line in file:lines() do
    local split_line = stringx.split(line)

    if start_of_unigrams then
      if #split_line == 2 then
        local token = split_line[2]
        local log10_prob = tonumber(split_line[1])
        local log_prob = torch.log(torch.pow(10, log10_prob))
        if token ~= "<unk>" then -- ignore KenLM added "<unk>"          
          local token_id
          if final_vocab_map[token] ~= nil then
            -- ignore extra symbols added by KenLM
            token_id = final_vocab_map[token]
            token_id_to_unigram[token_id] = log_prob
          end
          
        end
      end -- if #split_line == 2 then
    elseif #split_line == 2 and split_line[1] == "ngram" then
      start_of_unigrams = true
    end
  end
  file:close()
    
  -- final check that all vocab items have an associated unigram probability
  for token, token_id in pairs(final_vocab_map) do
    if action_symbols_list[token_id] == nil then -- action symbols (i.e., <eos> are not used for future costs) 
      if bnp == 1 then
        assert(token_id_to_unigram[token_id] ~= nil, string.format("%s (%d) is missing a unigram probability", token, token_id))      
      else
        assert(token_id ~= sonp_id and token_id ~= eonp_id, "BNP symbols should not be present in this case.")
        assert(token_id_to_unigram[token_id] ~= nil, string.format("%s (%d) is missing a unigram probability", token, token_id))
      end
    end
  end

  return token_id_to_unigram
end

local function save_parse(vocab_map, parse_data, output_parse_archive_path_with_filename, output_parse_path_with_filename, eos_id)
  --[[
    Saves the parse data (as .t7 and as strings) to disk.
  --]]
  
  torch.save(output_parse_archive_path_with_filename, parse_data) 
  
  -- save strings:
  local function get_idx_to_vocab_map(vocab_map)
    local idx_to_vocab = {}
    for vocab,idx in pairs(vocab_map) do
      idx_to_vocab[idx] = vocab
    end
    return idx_to_vocab
  end
  local idx_to_vocab = get_idx_to_vocab_map(vocab_map)
  
  local file = io.open(output_parse_path_with_filename, "w")  
  for i=1,parse_data:size(1) do
    if parse_data[i] == 0 then break end
    if parse_data[i] == eos_id then 
      file:write("\n") 
    else
      file:write(idx_to_vocab[parse_data[i]] .. " ")
    end
  end 
  file:close()
  print("Completed saving parse string")
  
end

local function save_scores(scores_data, output_parse_scores_with_filename)
  --[[
    Saves the scores assigned to the top beam for each sentence.
  --]]
  
  local file = io.open(output_parse_scores_with_filename, "w")
  for i=1,scores_data:size(1) do
    if scores_data[i] == 0 then break end
    file:write(scores_data[i] .. "\n")
  end
  
  file:close()
  
end

return {init_vocab_with_existing_vocab_map_and_existing_vocab_idx=init_vocab_with_existing_vocab_map_and_existing_vocab_idx,
        init_vocab_with_action_symbols=init_vocab_with_action_symbols,
        get_vocab_map_and_vocab_idx=get_vocab_map_and_vocab_idx,
        traindataset=traindataset,
        testdataset=testdataset,
        validdataset=validdataset,
        load_unigram_lm=load_unigram_lm,
        save_parse=save_parse,
        save_scores=save_scores}

