-- Train an LSTM language model

--[[ This is a modified version of https://github.com/wojzaremba/lstm/blob/master/main.lua,
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

-- needed for reading in the file containing labels
local stringx = require('pl.stringx')
local file = require('pl.file')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an LSTM language model')
cmd:text()
cmd:text('Options')

---- data
cmd:option('-data_dir','data/','data directory. Should contain input_train_filename, input_valid_filename, input_test_filename')
cmd:option('-input_train_filename','','Training filename (required)')
cmd:option('-input_valid_filename','','Valdation filename (required)')
cmd:option('-input_test_filename','','Test filename (Optional)')

---- data preprocessing options
cmd:option('-perform_text_preprocessing_int',1,'Int option. If 0, no preprocessing is performed. If 1, newline "\\n" is replaced with "<eos>".')
cmd:option('-run_test_on_valid',0,'Int option. If 1, the perplexity of the validation file (treating each symbol as a natural language token) is run.')
cmd:option('-run_test_on_tokens',0,'Int option. If 1, the perplexity of the test file (treating each symbol as a natural language token) is run. \z
   If 1, -input_test_filename must be provided.')

cmd:option('-checkpoint_dir', 'checkpoint', 'output directory in which to write checkpoints')
cmd:option('-savefile','lstm','Filename identifier to use for checkpoints. Will be inside checkpoint_dir/. default: lstm \z
  The filename will appear as lm_[IDENTIFIER]_[iteration|final].t7.')
cmd:option('-checkpoint_after_these_epochs','','Comma-separated string of epoch numbers after which to save a checkpoint of the current model.')

---- GPU
cmd:option('-gpuid',1,'GPUID to use by cutorch.setDevice(). Default is 1.')

---- RNN options
cmd:option('-seq_length',35,'RNN sequence length.')
cmd:option('-model_group',2, 'Groups of model parameters: 0 for small; 1 for large; 2, default (for settings used in paper). You \z
  may observe slightly improved results using the settings of 3 relative to the settings of 2.')
cmd:option('-batch_size',20, 'Batch size.')


-- parse input params
opt = cmd:parse(arg)

require('base')
local input_data_module = require('datastack')


local params
if opt.model_group == 0 then
  params = {layers=2,
                  decay=2,
                  rnn_size=200,
                  dropout=0,
                  init_weight=0.1,
                  lr=1,
                  max_epoch=4,
                  max_max_epoch=13,
                  max_grad_norm=5}
elseif opt.model_group == 1 then
  params = {layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
elseif opt.model_group == 2 then
  params = {layers=2,
                decay=1.5,
                rnn_size=650,
                dropout=0.50,
                init_weight=0.02,
                lr=1,
                max_epoch=10,
                max_max_epoch=30,
                max_grad_norm=5}
elseif opt.model_group == 3 then
  params = {layers=2,
                decay=1.2,
                rnn_size=650,
                dropout=0.50,
                init_weight=0.05,
                lr=1,
                max_epoch=6,
                max_max_epoch=39,
                max_grad_norm=5}
end

-- Set RNN options:
params.seq_length = opt.seq_length
params.batch_size = opt.batch_size

print("Checking command line arguments")

-- make sure output directory exists
g_path_exists_or_exit(opt.checkpoint_dir)

local input_train_path_with_filename = path.join(opt.data_dir, opt.input_train_filename) 
local input_valid_path_with_filename = path.join(opt.data_dir, opt.input_valid_filename) 
local input_test_path_with_filename
if opt.run_test_on_tokens == 1 then
  input_test_path_with_filename = path.join(opt.data_dir, opt.input_test_filename)
end

-- make sure input files exists
g_path_exists_or_exit(input_train_path_with_filename)
g_path_exists_or_exit(input_valid_path_with_filename)
if opt.run_test_on_tokens == 1 then
  g_path_exists_or_exit(input_test_path_with_filename)
end

print("Done checking command line arguments")

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test, state_valid_formatted_as_test
local model = {}
local paramx, paramdx

local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates)
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1]) -- for layer 1, this is x; for layer 2, this is the h from below (i.e., h_t)
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h -- in case of 2 layers, i[2] holds output/top h; i.e., i[params.layers]
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})

  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

local function setup()
  print("Creating an RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  print("Total number of parameters: " .. paramx:nElement())
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s})) -- ignoring output multinomial distribution here, pred_dist
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean() -- mean across seq_length of NLL's across this batch
end

local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s}, {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)  -- why the -1 here? since y at i + 1 in the data matrix
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1] -- perp_temp[1] should be a single number for the NLL for a single word (b/c in test, columns are duplicated)
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

local function run_test_on_valid()
  reset_state(state_valid_formatted_as_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_valid_formatted_as_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_valid_formatted_as_test.data[i]
    local y = state_valid_formatted_as_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1] -- perp_temp[1] should be a single number for the NLL for a single word (b/c in test, columns are duplicated)
    g_replace_table(model.s[0], model.s[1])
  end
  print("Perplexity on validation formatted as test : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end


local function save_checkpoint(iteration_id_str)
  --[[
  All structures, sans the data, needed for run_test() are saved to opt.checkpoint_dir/opt.savefile
  
  The trailing string indicates whether the file is a training or final checkpoint.
  
  --]]
  print("Saving a checkpoint of the model.")
   
  local savefile = string.format('%s/lm_%s_%s.t7', opt.checkpoint_dir, opt.savefile, iteration_id_str)
  
  print('saving checkpoint to ' .. savefile)
  local checkpoint = {}
  checkpoint.model = model
  -- save parameters
  checkpoint.params = params
  -- save command line arguments
  checkpoint.opt = opt

  torch.save(savefile, checkpoint)
        
end


local function save_vocab_map_and_vocab_idx(final_vocab_map, final_vocab_idx)
  local vocab_map_filepath = path.join(opt.checkpoint_dir, 'vocab_map.t7')
  local vocab_idx_filepath = path.join(opt.checkpoint_dir, 'vocab_idx.t7')
  
  torch.save(vocab_map_filepath, final_vocab_map)
  torch.save(vocab_idx_filepath, final_vocab_idx)
  
end


local function main()
  
  g_init_gpu(opt.gpuid)
  
  local action_symbols = "<eos>" -- the current trainer and decoder assume this constant; shift-reduce actions are not supported
  local eos_id = 1 -- index of the <eos> token in the vocabulary
  
  print("opt.eos_id: " .. eos_id)
  
  num_action_symbols = input_data_module.init_vocab_with_action_symbols(action_symbols)
  state_train = {data=transfer_data(input_data_module.traindataset(params.batch_size, opt.perform_text_preprocessing_int, input_train_path_with_filename))}
  state_valid =  {data=transfer_data(input_data_module.validdataset(params.batch_size, opt.perform_text_preprocessing_int, input_valid_path_with_filename))}
  if input_test_path_with_filename ~= nil then
    state_test =  {data=transfer_data(input_data_module.testdataset(params.batch_size, opt.perform_text_preprocessing_int, input_test_path_with_filename))}
  end
  
  local final_vocab_map, final_vocab_idx = input_data_module.get_vocab_map_and_vocab_idx()
  params.vocab_size = final_vocab_idx
  print("The total size of the vocab (including eos) used for the LookupTable is " .. params.vocab_size)
  save_vocab_map_and_vocab_idx(final_vocab_map, final_vocab_idx)
  
  local checkpoint_after_these_epochs = stringx.split(opt.checkpoint_after_these_epochs, ",")
  
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_train)    
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      if epoch > params.max_epoch then
          params.lr = params.lr / params.decay
      end
      for _, epoch_id in ipairs(checkpoint_after_these_epochs) do
        local epoch_id_str = stringx.strip(epoch_id)
        if torch.floor(epoch) == tonumber(epoch_id_str) then
            local iteration_id_str = 'epoch' .. g_f3(epoch_id_str) .. '_trainperp' .. g_f3(torch.exp(perps:mean()))
            save_checkpoint(iteration_id_str)    
        end
      end
      
    end
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  if opt.run_test_on_tokens == 1 then
    run_test()
  end
  if opt.run_test_on_valid == 1 then
    state_valid_formatted_as_test =  {data=transfer_data(input_data_module.testdataset(params.batch_size, opt.perform_text_preprocessing_int, input_valid_path_with_filename))}
    run_test_on_valid()
  end
  
  save_checkpoint("final")
  print("Training is over.")
end

main()
