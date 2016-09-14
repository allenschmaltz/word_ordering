-- Version 0.2
-- Generate/re-order with a lstm language model
        
local stringx = require('pl.stringx')


local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')


cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate/re-order with an lstm language model')
cmd:text()
cmd:text('Options')

---- data
cmd:option('-data_dir','data/','data directory. Should contain input_test_filename')
cmd:option('-input_test_filename','test.txt','Test filename, which should contain one shuffled sentence per line.')

---- data preprocessing options
cmd:option('-perform_text_preprocessing_int',1,'Int option. If 1, newline "\\n" is replaced with "<eos>". If 0, no preprocessing is performed.')

cmd:option('-checkpoint_dir', 'checkpoint', 'output directory from which to load the checkpoint')
cmd:option('-checkpoint_filename','lstm','filename of the checkpoint to load in checkpoint_dir')

---- logging options
cmd:option('-print_every',1000,'Number of natural language tokens between prining log of progress.')

---- Word-ordering options
cmd:option('-base_nps',1,'If 1, the data should include symbols indicating the start and end of base noun-phrases (BNPs). BNP symbols are treated as words. If 0, all tokens are considered independently and no bnp symbols should be present in the data.')

cmd:option('-base_np_symbols','<sonp>,<eonp>','Comma delimited string containing the symbols marking the start and end of base NPs. Default: <sonp>,<eonp>. (Only applicable with -base_nps=1.)')

cmd:option('-beam_size',0,'Size of beam. 0 runs the non-beam decoder.')

-- Future cost options
cmd:option('-unigram_lm_path_with_filename','', 'Location of an ARPA file for a unigram LM for future costs. If ommitted, \z
  no future costs are calculated.')

---- Output files
cmd:option('-output_dir', 'output_parse_and_scores', 'Output directory in which to save the parse and scores')
cmd:option('-output_parse_filename','parse','File in which to save the re-ordered output in -output_dir. Text is saved \z
  as .txt and torch object saved as .t7.')
cmd:option('-output_scores_filename','scores.txt','File in which to save scores (1 per line) in -output_dir (in txt format).')


---- GPU
cmd:option('-gpuid',1,'GPUID to use by cutorch.setDevice(). Default is 1.')


-- parse input params (note that local_opt are those provided to this script; opt is loaded from the checkpoint)
local local_opt = cmd:parse(arg)

require('base')
local input_data_module = require('datastack')


local state_test, params, opt
local model = {}
local action_symbols_list
local final_vocab_map, final_vocab_idx

local function transfer_data(x)
  return x:cuda()
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function copy_table_with_tensors(from_existing_table)
  local to_new_table = {}
  for i = 1, #from_existing_table do
    to_new_table[i] = from_existing_table[i]:clone()
  end
  return to_new_table
end

local function get_q_multinomial_dist()
  local q_multinomial_dist
  model.rnns[1]:apply(function(m) 
                        if torch.type(m) == "nn.LogSoftMax" then 
                          q_multinomial_dist = torch.exp(m.output:index(1,torch.LongTensor{1})) -- all rows in test are identical, so only retaining the 1st; note that this is new storage (not a new view)
                        end 
                      end)        
  return q_multinomial_dist
end

local function get_full_q_multinomial_dist()
  local q_multinomial_dist
  model.rnns[1]:apply(function(m) 
                        if torch.type(m) == "nn.LogSoftMax" then 
                          q_multinomial_dist = torch.exp(m.output)
                        end 
                      end)        
  return q_multinomial_dist
end

local function copy_table(original_structure)
  if type(original_structure) ~= "table" then
    return original_structure
  end
  local new_table = {}
  for k,v in pairs(original_structure) do 
    new_table[copy_table(k)] = copy_table(v) 
  end
  return new_table
end
    
local function copy_itable(original_structure)
  if type(original_structure) ~= "table" then
    return original_structure
  end
  local new_table = {}
  for k,v in ipairs(original_structure) do 
    new_table[copy_table(k)] = copy_table(v) 
  end
  return new_table
end


local function batch_advance_full(states, q_multinomial_dists, action, current_beam_size, target_token)
  local out_q_multinomial_dists = q_multinomial_dists 
  local scores = transfer_data(torch.zeros(current_beam_size, 1))
  
  -- an initial copy to set dimensions of model.s[0] to states (and to avoid clobbering states)
  model.s[0] = copy_table_with_tensors(states)
  
  local out_states
  for w_id,w in ipairs(action) do -- action is a table of token_ids

    scores = scores + torch.log(out_q_multinomial_dists:narrow(2,w,1))
    local current_sym = transfer_data(torch.zeros(current_beam_size):fill(w))
    _, out_states = unpack(model.rnns[1]:forward({current_sym, target_token, model.s[0]})) 
  
    g_replace_table(model.s[0], out_states)

    out_q_multinomial_dists = get_full_q_multinomial_dist()
  end

  return scores, out_states, out_q_multinomial_dists
end

local function Hypothesis(score, last_action, bow, future_score, state, last_beam)
  --[[
    This function serves to encaspulate an hypothesis datastructure, which is a table with the following properties:
  
    score
    last_action
    bow
    state  {['state']=RNN_STATE, ['q_multinomial_dist']=OUTPUT_SOFTMAX_DISTRIBUTION}
    last_beam  -- index backpointer for recovering the previous actions
  --]]
  local hypothesis = {}      
  hypothesis.score = score
  hypothesis.last_action = last_action
  hypothesis.bow = bow
  hypothesis.future_score = future_score
  hypothesis.state = state
  hypothesis.last_beam = last_beam  
  return hypothesis
end
  
local function hypothesis_comparison_future(a,b)
  -- Hypotheses are sorted from highest to lowest by score+future_score
  return a.score + a.future_score > b.score + b.future_score
end

local function future(idx_to_action, idx_to_freq, futurelm)
  --[[
    idx_to_freq should already have been decremented with the current action_idx
  --]]
    local score = 0.0
    if next(futurelm) ~= nil then -- futurelm is {} when no unigram lm is provided
      for action_idx, freq in ipairs(idx_to_freq) do
        if freq ~= 0 then
          local action = idx_to_action[action_idx]

          for _,w in ipairs(action) do
            assert(futurelm[w] ~= nil, "Unigram probabilities should be present for all tokens")
            score = score + futurelm[w] * freq
          end
        end
        
      end
    end
    return score
end

local function generate(lm, idx_to_action, idx_to_freq, beam_size, futurelm)

  local n = 0
  for idx,action in ipairs(idx_to_action) do
    n = n + idx_to_freq[idx]*#action
  end

  local start_state = {}
  start_state.state = {}
  for d = 1, 2 * params.layers do
    start_state.state[d] = lm.current_state[d][1] -- here, all of the rows should be identical, so just taking 1st
  end
  start_state.q_multinomial_dist = lm.q_multinomial_dist -- this is the distribution from forwarding the initial <eos> symbol
  
  local bow = idx_to_freq
  local order = {}
  local beams = {}
  beams[1] = {[1]=Hypothesis(0.0, {}, copy_itable(bow), future(idx_to_action, bow, futurelm), start_state, -1)}

  for i=2,n+1 do
    beams[i] = {}
  end
    
  for i=1,n do 
    local states = {} 
    local current_beam_size = #beams[i] + 1 -- extra dim to avoid dim error (with squeeze) when size == 1
    local target_token_holder = transfer_data(torch.zeros(current_beam_size):fill(opt.eos_id))
    local q_multinomial_dists = transfer_data(torch.zeros(current_beam_size, lm.q_multinomial_dist:size(2)))

    for d = 1, 2 * params.layers do
      local state_zero = transfer_data(torch.zeros(current_beam_size, params.rnn_size))
      for j,hyp in ipairs(beams[i]) do
        state_zero[j] = hyp.state.state[d] -- each vector corresponds to the history associated with beam_idx
        if d == 1 then
          q_multinomial_dists[j] = hyp.state.q_multinomial_dist
        end
        
      end
      states[d] = state_zero
    end
    
    for action_idx, _ in ipairs(bow) do
      local action_appears_in_at_least_one_hyp = false
      for j,hyp in ipairs(beams[i]) do
        if hyp.bow[action_idx] ~= 0 then
          action_appears_in_at_least_one_hyp = true
          break
        end
      end
      
      -- Only batch forward an action for which at least 1 hypothesis contains the action
      if action_appears_in_at_least_one_hyp then
        local action = idx_to_action[action_idx]

        local scores, out_states, out_q_multinomial_dists = batch_advance_full(states, q_multinomial_dists, action, current_beam_size, target_token_holder)

        -- Add to beam
        local ni = i + #action

        for j,hyp in ipairs(beams[i]) do
          local score = scores[j][1] 
          
          if hyp.bow[action_idx] ~= 0 then
            assert(beams[ni] ~= nil, "beams from 1 to n+1 should already be initilized")
            local new_bow = copy_itable(hyp.bow)
            new_bow[action_idx] = new_bow[action_idx] - 1
            local fscore = future(idx_to_action, new_bow, futurelm)            
            if (#beams[ni] < beam_size) or (hyp.score + score + fscore > beams[ni][#beams[ni]].score + beams[ni][#beams[ni]].future_score) then

              local new_state = {}
              new_state.state = {}

              for d = 1, 2 * params.layers do
                new_state.state[d] = out_states[d][j]:clone() 
              end
              new_state.q_multinomial_dist = out_q_multinomial_dists[j]:clone()

              local new_hyp = Hypothesis(hyp.score + score, action, new_bow, fscore, new_state, j)
              
              beams[ni][#beams[ni] + 1] = new_hyp
              table.sort(beams[ni], hypothesis_comparison_future)
              if #beams[ni] > beam_size then
                -- remove the lowest scoring Hypothesis from the beam
                 table.remove(beams[ni], #beams[ni]) 
              end
              assert(#beams[ni] <= beam_size)
            end  -- if (#beams[ni] <= beam_size) or (hyp.score + score > beams[ni][#beams[ni]].score) then
          end -- if hyp.bow[action_idx] ~= 0 then
          
        end -- for j,hyp in ipairs(beams[i]) do
      end -- if action_appears_in_at_least_one_hyp then
    end -- for action_idx, _ in ipairs(bow) do
    
  end -- for i=1,n do
    
  
  local cur = n + 1 -- since the first index is the initial/null
  local pos = 1 -- index into the Hypothesis for a beam of a given sequence length
  local sentence_token_index = n
  local order = torch.zeros(n)
  local final_score = beams[cur][pos].score
  while(true) do
    if cur <= 1 then break end
    for j=#beams[cur][pos].last_action,1,-1 do
      order[sentence_token_index] = beams[cur][pos].last_action[j]
      sentence_token_index = sentence_token_index - 1
    end
    local old_cur = cur
    cur = cur - #beams[cur][pos].last_action
    pos = beams[old_cur][pos].last_beam
  end
  assert(not torch.any(torch.eq(order,0)), "Not all of the words in the sentence bag were covered.")
  
  return order, final_score
end


local function generate_main(eos_id, sonp_id, eonp_id, beam_size, token_id_to_unigram)

  local print_every = local_opt.print_every
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local parse_data = transfer_data(torch.zeros(state_test.data:size(1))) 
  local parse_data_ctr = 1
  local len = state_test.data:size(1)
  
  -- re-initialize initial states to correspond to the beam_size
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(local_opt.beam_size, params.rnn_size))
    model.s[0][d] = transfer_data(torch.zeros(local_opt.beam_size, params.rnn_size))
  end
  
  local current_state = copy_table_with_tensors(model.start_s)
  
  local final_scores = transfer_data(torch.zeros(state_test.data:size(1))) 
  local final_scores_ctr = 1
  
  local sentence_ctr = 0
  local start_i = 1 -- first natural language token in the data
  local end_i = -1 -- index including eos
   
  local target_token = state_test.data[1]:clone():fill(final_vocab_map[action_symbols_list[eos_id]]) -- this is just a holder; actual distribution is pulled from softmax
  -- start by pushing through <eos>
  local current_sym = state_test.data[1]:clone():fill(final_vocab_map[action_symbols_list[eos_id]])
  g_replace_table(model.s[0], current_state)
  _, current_state = unpack(model.rnns[1]:forward({current_sym, target_token, model.s[0]})) 


  while(true) do -- loop through all sentences
    local seen_sonp = false
    local total_tokens_in_sent_ctr = 0
  
    local phrase_set = {} -- set of word phrases (no duplicates) {[phrase string]=frequency}
    local phrase_string = ""
    
    for k = start_i,len do -- build phrase_set:
      if state_test.data[k][1] == eos_id then -- the sentences are complete after hitting the eos symbol
        end_i = k
        assert(string.len(phrase_string) == 0)
        if true then break end
      elseif state_test.data[k][1] == eonp_id then  -- end of a base NP
        assert(seen_sonp)
        assert(string.len(phrase_string) > 0)
        -- add the ending base NP symbol:
        phrase_string = phrase_string .. " " .. eonp_id
        total_tokens_in_sent_ctr = total_tokens_in_sent_ctr + 1
        if phrase_set[phrase_string] then
          phrase_set[phrase_string] = phrase_set[phrase_string] + 1
        else
          phrase_set[phrase_string] = 1
        end
        phrase_string = ""
        seen_sonp = false
      elseif state_test.data[k][1] == sonp_id then  -- start of a base NP
        assert(string.len(phrase_string) == 0)
        assert(not seen_sonp)
        seen_sonp = true
        -- add the starting base NP symbol:
        phrase_string = "" .. sonp_id
        total_tokens_in_sent_ctr = total_tokens_in_sent_ctr + 1
      else 
        total_tokens_in_sent_ctr = total_tokens_in_sent_ctr + 1
        if seen_sonp then -- inside base NP
          assert(string.len(phrase_string) ~= 0, "The start base NP should already have been added")
          phrase_string = phrase_string .. " " .. state_test.data[k][1]
        else -- a single token outside of a base NP
          assert(string.len(phrase_string) == 0)
          if phrase_set["" .. state_test.data[k][1]] then
            phrase_set["" .. state_test.data[k][1]] = phrase_set["" .. state_test.data[k][1]] + 1
          else
            phrase_set["" .. state_test.data[k][1]] = 1
          end
        end
      end
    end
    if sentence_ctr % print_every == 0 then
      print("currently processing sentence " .. sentence_ctr)
    end
    assert(string.len(phrase_string) == 0)
    assert(not seen_sonp)
    assert(state_test.data[start_i][1] > eos_id)
    assert(end_i ~= -1)
    assert(state_test.data[end_i][1] == eos_id)
    
    local idx_to_action = {} -- {[idx]={[1]=vocab_id1, [2]=vocab_id2, ...},...}}
    local idx_to_freq = {} -- {[idx]=frequency,...}}
    
    local idx_ctr = 1
    for phrase_string, phrase_freq in pairs(phrase_set) do
      local action_strings = stringx.split(phrase_string)
      local action = {}
      for token_index,token_string in ipairs(action_strings) do
        action[token_index] = tonumber(token_string)
      end
                
      idx_to_action[idx_ctr] = action
      idx_to_freq[idx_ctr] = phrase_freq
      idx_ctr = idx_ctr + 1
    end
    
    local sent_len = end_i - start_i -- number of symbols other than <eos> (this includes the <sonp>,<eonp> symbols)
       
    local lm = {}    
    lm.current_state = current_state
    lm.q_multinomial_dist  = get_q_multinomial_dist()
  
    local order, final_score = generate(lm, idx_to_action, idx_to_freq, beam_size, token_id_to_unigram)
    
    final_scores[final_scores_ctr] = final_score
    final_scores_ctr = final_scores_ctr + 1
    
    -- For the word ordering task, reset after each sentence:
    model.s[0] = copy_table_with_tensors(model.start_s)
    
    -- update the running parse
    parse_data[{{parse_data_ctr,parse_data_ctr+total_tokens_in_sent_ctr-1}}] = order
    parse_data_ctr = parse_data_ctr + total_tokens_in_sent_ctr
    
    -- push through <eos>
    local current_sym = current_sym:clone():fill(final_vocab_map[action_symbols_list[eos_id]])
    _, current_state = unpack(model.rnns[1]:forward({current_sym, target_token, model.s[0]}))
    parse_data[parse_data_ctr] = current_sym[1]
    parse_data_ctr = parse_data_ctr + 1
    
    -- update counters in preparation for next sentence
    sentence_ctr = sentence_ctr + 1
    start_i = end_i + 1
    end_i = -1
    if start_i > len then break end

    cutorch.synchronize()
    collectgarbage()
          
  end

  g_enable_dropout(model.rnns)
  return parse_data, final_scores
end -- end of generate_main()


local function load_checkpoint_and_vocab()

  local checkpoint_path_with_filename = path.join(local_opt.checkpoint_dir, local_opt.checkpoint_filename)
  print("Loading a checkpoint of the model:" .. checkpoint_path_with_filename)

  local checkpoint = torch.load(checkpoint_path_with_filename)
  model = checkpoint.model
  params = checkpoint.params
  opt = checkpoint.opt
  
  local vocab_map_filepath = path.join(local_opt.checkpoint_dir, 'vocab_map.t7')
  local vocab_idx_filepath = path.join(local_opt.checkpoint_dir, 'vocab_idx.t7')
  
  final_vocab_map = torch.load(vocab_map_filepath)
  final_vocab_idx = torch.load(vocab_idx_filepath)
  
  -- this decoder assume the following two constants
  opt.action_symbols = "<eos>" -- shift-reduce actions are not supported in this version
  opt.eos_id = 1 -- index of the <eos> token in the vocabulary
  
  action_symbols_list = stringx.split(opt.action_symbols,",")
  
  print("Checkpoint opt:")
  print(opt)
  
  print("Checkpoint params:")
  print(params)
  
  print("Checkpoint vocab size: " .. final_vocab_idx)
  
  print("Parameters provided for this script:")
  print(local_opt)
  
end


local function main()  

  g_init_gpu(local_opt.gpuid)
  
  -- check that the output dir exists and the provided output files do not exist:
  g_path_exists_or_exit(local_opt.output_dir)
  local output_parse_archive_path_with_filename = path.join(local_opt.output_dir, local_opt.output_parse_filename .. ".t7")
  local output_parse_path_with_filename = path.join(local_opt.output_dir, local_opt.output_parse_filename .. ".txt")
  local output_parse_scores_with_filename = path.join(local_opt.output_dir, local_opt.output_scores_filename)
  g_path_does_not_exist_or_exit(output_parse_archive_path_with_filename)
  g_path_does_not_exist_or_exit(output_parse_path_with_filename)
  g_path_does_not_exist_or_exit(output_parse_scores_with_filename)
    
  load_checkpoint_and_vocab()
  input_data_module.init_vocab_with_existing_vocab_map_and_existing_vocab_idx(final_vocab_map, final_vocab_idx)
  
  local sonp_id, eonp_id  
  if local_opt.base_nps == 0 then
    print("Running word-ordering beam decoder without marked base NPs.")
    sonp_id = final_vocab_idx + 100
    eonp_id = final_vocab_idx + 200
  elseif local_opt.base_nps == 1 then
    print("Running word-ordering beam decoder with marked base NPs.")
    local base_np_symbols_list = stringx.split(local_opt.base_np_symbols,",")  
    sonp_id = final_vocab_map[base_np_symbols_list[1]]
    eonp_id = final_vocab_map[base_np_symbols_list[2]]
  end
      
  params.batch_size = local_opt.beam_size
  
  local input_test_path_with_filename = path.join(local_opt.data_dir, local_opt.input_test_filename)
  
  state_test =  {data=transfer_data(input_data_module.testdataset(params.batch_size, local_opt.perform_text_preprocessing_int, input_test_path_with_filename))}
  
  print("The total size of the vocab used for the LookupTable is " .. params.vocab_size)
  
  local token_id_to_unigram = {}
  if string.len(local_opt.unigram_lm_path_with_filename) > 0 then
    g_path_exists_or_exit(local_opt.unigram_lm_path_with_filename)
    token_id_to_unigram = input_data_module.load_unigram_lm(local_opt.unigram_lm_path_with_filename, final_vocab_map, action_symbols_list, local_opt.base_nps, sonp_id, eonp_id)
    print("Future costs are being used.")
  else
    print("Future costs are NOT being used.")
  end
  
  reset_state(state_test)

  timer = torch.Timer()

  local parse_data
  local final_scores
 
  parse_data, final_scores = generate_main(opt.eos_id, sonp_id, eonp_id, local_opt.beam_size, token_id_to_unigram)
 
  print('Time elapsed for beam decoding: ' .. timer:time().real/60 .. ' minutes')
  
  if parse_data ~= nil then
    print("saving parse data")
    input_data_module.save_parse(final_vocab_map, parse_data, output_parse_archive_path_with_filename, output_parse_path_with_filename, opt.eos_id)
  else
    print("parse_data is nil and is not being saved")
  end
 
  if final_scores ~= nil then
    print("saving scores data")
    input_data_module.save_scores(final_scores, output_parse_scores_with_filename)
  else
    print("final_scores is nil and is not being saved")
  end
  
  print("Re-ordering is complete.")
end

main()


