--[[

  Sentiment classification using a Binary Tree-LSTM with Attention.

--]]
local AttentionTreeLSTMSentiment = torch.class('treelstm.AttentionTreeLSTMSentiment')

function AttentionTreeLSTMSentiment:__init(config)
  self.mem_dim           = config.mem_dim           or 150
  self.learning_rate     = config.learning_rate     or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.1
  self.reg               = config.reg               or 1e-4
  self.structure         = config.structure         or 'attention'
  self.fine_grained      = (config.fine_grained == nil) and true or config.fine_grained
  self.batch_size        = config.batch_size or 25
  self.max_nodes         = config.max_nodes
  self.max_train         = config.max_train
  self.max_dev           = config.max_dev
  self.max_test          = config.max_test
  self.accuracy          = 0.0
  self.epochs            = config.epochs

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.in_zeros = torch.zeros(self.emb_dim)
  self.num_classes = self.fine_grained and 5 or 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- negative log likelihood optimization objective
  self.criterion = nn.ClassNLLCriterion()

  local treelstm_config = {
    in_dim  = self.emb_dim,
    mem_dim = self.mem_dim,
    max_nodes = self.max_nodes,
  }
  local lstm_decoder_config = {
    mem_dim = self.mem_dim
  }

  self.treelstm = treelstm.ChildSumTreeLSTM(treelstm_config)
  self.linear_layer = self:linear_layer()
  self.lstm_decoder = treelstm.LSTMDecoder(lstm_decoder_config)

  self.master_sentiment_decoder = self:sentiment_decoder()
  self.sentiment_decoders = {}

  local modules = nn.Parallel()
    :add(self.treelstm)
    :add(self.linear_layer)
    :add(self.lstm_decoder)
    :add(self.master_sentiment_decoder)
  self.params, self.grad_params = modules:getParameters()
end

function AttentionTreeLSTMSentiment:linear_layer()
  local rep = nn.Identity()()
  local inputs = {rep}
  local linear = nn.Linear(self.mem_dim * self.max_nodes, self.mem_dim * self.max_nodes)(inputs)
  return nn.gModule(inputs, {linear})
end

function AttentionTreeLSTMSentiment:sentiment_decoder()
  local input_dim = self.mem_dim
  local rep = nn.Identity()()
  local inputs = {rep}
  local logprob = nn.LogSoftMax()(nn.Linear(input_dim, self.num_classes)(inputs))
  local outp= nn.gModule(inputs, {logprob})

  if self.master_sentiment_decoder then
    share_params(outp, self.master_sentiment_decoder)
  end

  return outp
end

function AttentionTreeLSTMSentiment:train(dataset, test_dataset)
  self.treelstm:training()
  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  dataset.size = self.max_train

  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local sent = dataset.sents[idx]
        local tree = dataset.trees[idx]

        local inputs = self.emb:forward(sent)
        local tree_states = self.treelstm:forward(tree, inputs, true)
        tree_states = torch.view(tree_states, 1, self.max_nodes * self.mem_dim)
        local output_linear = self.linear_layer:forward(tree_states)
        output_linear = torch.view(output_linear, self.max_nodes, self.mem_dim)
        local output_decoder = self.lstm_decoder:forward(output_linear)
        local log_probs = {}
        for i = 1, #output_decoder do
          local sentiment_decoder = self.sentiment_decoders[i]
          if sentiment_decoder == nil then
            sentiment_decoder = self:sentiment_decoder()
            self.sentiment_decoders[i] = sentiment_decoder
          end
          log_probs[i] = sentiment_decoder:forward(output_decoder[i])
        end
        local subtrees = tree:depth_first_preorder()
        local predictions
        local phrase_loss = 0.0
        local obj_grad = {}
        local log_probs_grad={}
        for i = 1, #subtrees do
          predictions = log_probs[i]
          if subtrees[i].gold_label ~= nil then
            phrase_loss = phrase_loss + self.criterion:forward(predictions, subtrees[i].gold_label)
            obj_grad[i] = self.criterion:backward(predictions, subtrees[i].gold_label)
          else
            obj_grad[i] = 0.0 * self.criterion:backward(predictions, 3)
          end
          local sentiment_decoder = self.sentiment_decoders[i]
          log_probs_grad[i] = sentiment_decoder:backward(output_decoder[i], obj_grad[i])
        end

        loss = loss + phrase_loss
        local decoder_grad = self.lstm_decoder:backward(output_linear, torch.cat(log_probs_grad))
        decoder_grad = torch.view(decoder_grad, 1, self.max_nodes * self.mem_dim)
        local linear_grad = self.linear_layer:backward(tree_states, decoder_grad)
        linear_grad = torch.view(linear_grad, self.max_nodes, self.mem_dim)
        local input_grad = self.treelstm:backward(tree, inputs, {zeros, zeros}, linear_grad)
        self.emb:backward(sent, input_grad)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
    self.emb:updateParameters(self.emb_learning_rate)
    collectgarbage()
  end
  xlua.progress(dataset.size, dataset.size)
end

function AttentionTreeLSTMSentiment:predict(tree, sent)
  self.treelstm:evaluate()
  local prediction
  local inputs = self.emb:forward(sent)
  self.treelstm:forward(tree, inputs)
  local output = tree.output
  if self.fine_grained then
    prediction = argmax(output)
  else
    prediction = (output[1] > output[3]) and 1 or 3
  end
  self.treelstm:clean(tree)
  return prediction
end

function AttentionTreeLSTMSentiment:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.trees[i], dataset.sents[i])
  end
  return predictions
end

function table.slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
end


function AttentionTreeLSTMSentiment:get_final_accuracy(dataset, type)
  local total=0
  local correct=0
  if (type == 'train') then
    dataset.size = self.max_train
  elseif (type == 'dev') then
    dataset.size = self.max_dev
  elseif (type == 'test') then
    dataset.size = self.max_test
  end

  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local inputs = self.emb:forward(dataset.sents[i])
    local tree_states = self.treelstm:forward(dataset.trees[i], inputs, true)
    tree_states = torch.view(tree_states, 1, self.max_nodes * self.mem_dim)
    local output_linear = self.linear_layer:forward(tree_states)
    output_linear = torch.view(output_linear, self.max_nodes, self.mem_dim)
    local output_decoder = self.lstm_decoder:forward(output_linear)
    local log_probs = {}
    for i = 1, #output_decoder do
      local sentiment_decoder = self.sentiment_decoders[i]
      log_probs[i] = sentiment_decoder:forward(output_decoder[i])
    end
    local subtrees = dataset.trees[i]:depth_first_preorder()
    for i = 1, #subtrees do
      local output = log_probs[i]
      local prediction
      if self.fine_grained then
        prediction = argmax(output)
      else
        prediction = (output[1] > output[3]) and 1 or 3
      end
      if subtrees[i].gold_label~=nil and prediction == subtrees[i].gold_label then
        correct = correct + 1
      end
      if subtrees[i].gold_label ~= nil then
        total = total + 1
        --printf("%d - %d\n", prediction, subtrees[i].gold_label)
        --print(log_probs[i])
      end
    end
  --print(correct)
  --print("^Correct\n")
  --print(correct/total)
  end
  --print("Accuracy .... : ")
  --print(correct/total)
  self.accuracy = correct/total
  return self.accuracy
end

function AttentionTreeLSTMSentiment:get_accuracy(dataset)
  local total=0
  local correct=0

  for i = 1, dataset.size do
   xlua.progress(i, dataset.size)
    print(dataset.sents[i])
    local inputs = self.emb:forward(dataset.sents[i])
    local currentTree = dataset.trees[i]
       local ct
        local cc

    ct,cc=self:get_tree_accuracy(currentTree, inputs)
    total=total+ct
    correct=correct+cc
  --print(correct)
  --print("^Correct\n")
  --print(correct/total)
  end
  print("Accuracy .... : ")
  print(correct/total)
end

function AttentionTreeLSTMSentiment:get_tree_accuracy(tree, inputs)
    local correct=0
    local total=0
    self.treelstm:forward(tree, inputs, true)
    local output = tree.output
    local prediction
    if self.fine_grained then
       prediction = argmax(output)
    else
       prediction = (output[1] > output[3]) and 1 or 3
    end
    --print("Prediction: " )
    if prediction==tree.gold_label then
      --print("Prediction: " )
      --print(prediction)
      --print("\n")
      --print("GL\n")
      --print(tree.gold_label)
      correct=correct+1
    end
    total=total+1
    for i = 1, tree.num_children do
      if tree.children[i].gold_label~=nil then
       local ct
        local cc
       ct, cc =  self:get_tree_accuracy(tree.children[i], inputs)
      total=total+ct
      correct=correct+cc
       end
    end
    return total, correct
 end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function AttentionTreeLSTMSentiment:print_config()
  local num_params = self.params:size(1)
  local num_sentiment_params = self.master_sentiment_decoder:getParameters():size(1)
  print('ATTENTION')
  printf('%-25s = %d\n',   'epochs', self.epochs)
  printf('%-25s = %s\n',   'fine grained sentiment', tostring(self.fine_grained))
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sentiment_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %d\n', 'training size', self.max_train)
  printf('%-25s = %d\n', 'dev size', self.max_dev)
  printf('%-25s = %d\n', 'test size', self.max_test)
  printf('%-25s = %d\n', 'max nodes in Tree', self.max_nodes)
end

function AttentionTreeLSTMSentiment:save_results(path)
  local f = assert(io.open(path, "w"))
  io.output(f)
  local num_params = self.params:size(1)
  local num_sentiment_params = self.master_sentiment_decoder:getParameters():size(1)
  io.write('ATTENTION')
  io.write(string.format('%-25s = %d\n',   'epochs', self.epochs))
  io.write(string.format('%-25s = %s\n',   'fine grained sentiment', tostring(self.fine_grained)))
  io.write(string.format('%-25s = %d\n',   'num params', num_params))
  io.write(string.format('%-25s = %d\n',   'num compositional params', num_params - num_sentiment_params))
  io.write(string.format('%-25s = %d\n',   'word vector dim', self.emb_dim))
  io.write(string.format('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim))
  io.write(string.format('%-25s = %.2e\n', 'regularization strength', self.reg))
  io.write(string.format('%-25s = %d\n',   'minibatch size', self.batch_size))
  io.write(string.format('%-25s = %.2e\n', 'learning rate', self.learning_rate))
  io.write(string.format('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate))
  io.write(string.format('%-25s = %d\n', 'training size', self.max_train))
  io.write(string.format('%-25s = %d\n', 'dev size', self.max_dev))
  io.write(string.format('%-25s = %d\n', 'test size', self.max_test))
  io.write(string.format('%-25s = %d\n', 'max nodes in Tree', self.max_nodes))
  io.write(string.format('%-25s = %.8f\n', 'accuracy', self.accuracy))
  f:close()
end

function AttentionTreeLSTMSentiment:save(path)
  local config = {
    batch_size        = self.batch_size,
    emb_learning_rate = self.emb_learning_rate,
    emb_vecs          = self.emb.weight:float(),
    fine_grained      = self.fine_grained,
    learning_rate     = self.learning_rate,
    mem_dim           = self.mem_dim,
    reg               = self.reg,
    structure         = self.structure,
  }

  torch.save(path, {
    params = self.params,
    config = config,
  })
end

function AttentionTreeLSTMSentiment.load(path)
  local state = torch.load(path)
  local model = treelstm.AttentionTreeLSTMSentiment.new(state.config)
  model.params:copy(state.params)
  return model
end
