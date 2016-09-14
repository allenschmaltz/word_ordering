--[[ This is based on the code at https://github.com/wojzaremba/lstm/blob/master/base.lua,
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


function g_path_exists_or_exit(dir_or_file)
  if not path.exists(dir_or_file) then 
    print("The following file or directory does not exist: ")
    print(dir_or_file)
    print("Exiting.")
    
    os.exit()  
  end
end

function g_path_does_not_exist_or_exit(dir_or_file)
  if path.exists(dir_or_file) then 
    print("The following file or directory exists: ")
    print(dir_or_file)
    print("Exiting in order to avoid overwriting.")
    
    os.exit()  
  end
end

function g_disable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end

function g_enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_enable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
  end
end

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end

--function g_init_gpu(args)
--  local gpuidx = args
--  gpuidx = gpuidx[1] or 1
function g_init_gpu(gpuidx_arg)
  local gpuidx = 1
  if gpuidx_arg ~= nil then
    gpuidx = gpuidx_arg
  end
  
  print(string.format("Using %s-th gpu", gpuidx))
  cutorch.setDevice(gpuidx)
  g_make_deterministic(1)
end

function g_make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function g_f3(f)
  return string.format("%.3f", f)
end

function g_d(f)
  return string.format("%d", torch.round(f))
end
