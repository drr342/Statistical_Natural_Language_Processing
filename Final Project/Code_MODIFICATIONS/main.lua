--[[

  Tree-LSTM training script for sentiment classication on the Stanford
  Sentiment Treebank

--]]

require('..')

function accuracy(pred, gold)
  return torch.eq(pred, gold):sum() / pred:size(1)
end

-- read command line arguments
local args = lapp [[
Training script for sentiment classification on the SST dataset.
  -m,--model  (default constituency) Model architecture: [constituency, lstm, bilstm]
  -l,--layers (default 1)            Number of layers (ignored for Tree-LSTM)
  -d,--dim    (default 150)          LSTM memory dimension
  -e,--epochs (default 10)           Number of training epochs
  -b,--binary                        Train and evaluate on binary sub-task
  -t,--trainsize (default 0)
  -v,--devsize   (default 0)
  -s,--testsize  (default 0)
  -a,--batchsize (default 0)
  -x,--maxnodes  (default 100)
  -r,--reg       (default 1e-4)
  -t,--lr        (default 0.05)
  -w,--welr      (default 0.1)
]]

local model_name, model_class, model_structure
if args.model == 'constituency' then
  model_name = 'Constituency Tree LSTM'
  model_class = treelstm.TreeLSTMSentiment
elseif args.model == 'dependency' then
  model_name = 'Dependency Tree LSTM'
  model_class = treelstm.TreeLSTMSentiment
--------------------------------------------------------------------------------
elseif args.model == 'attention' then
  model_name = 'Attention Tree LSTM'
  model_class = treelstm.AttentionTreeLSTMSentiment
elseif args.model == 'encoder' then
  model_name = 'Encoder Tree LSTM '
  model_class = treelstm.EncoderTreeLSTMSentiment
--------------------------------------------------------------------------------
elseif args.model == 'lstm' then
  model_name = 'LSTM'
  model_class = treelstm.LSTMSentiment
elseif args.model == 'bilstm' then
  model_name = 'Bidirectional LSTM'
  model_class = treelstm.LSTMSentiment
end
model_structure = args.model
header(model_name .. ' for Sentiment Classification')

-- binary or fine-grained subtask
local fine_grained = not args.binary

-- directory containing dataset files
local data_dir = 'data/sst/'

-- load vocab
local vocab = treelstm.Vocab(data_dir .. 'vocab-cased.txt')

-- load embeddings
print('loading word embeddings')
local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = treelstm.read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

-- use only vectors in vocabulary (not necessary, but gives faster training)
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
  local w = string.gsub(vocab:token(i), '\\', '') -- remove escape characters
  if emb_vocab:contains(w) then
    vecs[i] = emb_vecs[emb_vocab:index(w)]
  else
    num_unk = num_unk + 1
    vecs[i]:uniform(-0.05, 0.05)
  end
end
print('unk count = ' .. num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

-- load datasets
print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local dependency = (args.model == 'dependency' or args.model == 'attention' or args.model == 'encoder')
local train_dataset = treelstm.read_sentiment_dataset(train_dir, vocab, fine_grained, dependency)
local dev_dataset = treelstm.read_sentiment_dataset(dev_dir, vocab, fine_grained, dependency)
local test_dataset = treelstm.read_sentiment_dataset(test_dir, vocab, fine_grained, dependency)

printf('num train = %d\n', train_dataset.size)
printf('num dev   = %d\n', dev_dataset.size)
printf('num test  = %d\n', test_dataset.size)

-- initialize model
local model = model_class{
  emb_vecs = vecs,
  structure = model_structure,
  fine_grained = fine_grained,
  num_layers = args.layers,
  mem_dim = args.dim,
  max_train = (args.trainsize ~= 0) and args.trainsize or train_dataset.size,
  max_dev = (args.devsize ~= 0) and args.devsize or dev_dataset.size,
  max_test = (args.testsize ~= 0) and args.testsize or test_dataset.size,
  batch_size = (args.batchsize ~= 0) and args.batchsize or 25,
  max_nodes = args.maxnodes,
  reg = args.reg,
  learning_rate = args.lr,
  emb_learning_rate = args.welr,
  epochs = args.epochs
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
header('Training model')
function table.slice(tbl, first, last, step)
  local sliced = {}

  for i = first or 1, last or #tbl, step or 1 do
    sliced[#sliced+1] = tbl[i]
  end

  return sliced
end
if lfs.attributes(treelstm.log) == nil then
  lfs.mkdir(treelstm.log)
end
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset, test_dataset)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  -- uncomment to compute train scores
  --[[
  local train_predictions = model:predict_dataset(train_dataset)
  local train_score = accuracy(train_predictions, train_dataset.labels)
  printf('-- train score: %.4f\n', train_score)
  --]]

  local dev_score = model:get_final_accuracy(dev_dataset, 'dev')
  --local dev_predictions = model:predict_dataset(dev_dataset)
  --local dev_score = accuracy(dev_predictions, dev_dataset.labels)
  printf('-- dev score: %.4f\n', dev_score)

  if dev_score > best_dev_score then
    best_dev_score = dev_score
    best_dev_model = model_class{
      emb_vecs = vecs,
      structure = model_structure,
      fine_grained = fine_grained,
      num_layers = args.layers,
      mem_dim = args.dim,
      max_train = (args.trainsize ~= 0) and args.trainsize or train_dataset.size,
      max_dev = (args.devsize ~= 0) and args.devsize or dev_dataset.size,
      max_test = (args.testsize ~= 0) and args.testsize or test_dataset.size,
      batch_size = (args.batchsize ~= 0) and args.batchsize or 25,
      max_nodes = args.maxnodes,
      reg = args.reg,
      learning_rate = args.lr,
      emb_learning_rate = args.welr,
      epochs = args.epochs
    }
    best_dev_model.params:copy(model.params)
    best_dev_model.emb.weight:copy(model.emb.weight)
    best_dev_model.accuracy = model.accuracy
    best_dev_model.sentiment_decoders = model.sentiment_decoders
  end
end
printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_score = best_dev_model:get_final_accuracy(test_dataset, 'test')
--local test_predictions = best_dev_model:predict_dataset(test_dataset)
printf('-- test score: %.4f\n', test_score)

--[[
-- create predictions and models directories if necessary
if lfs.attributes(treelstm.predictions_dir) == nil then
  lfs.mkdir(treelstm.predictions_dir)
end

if lfs.attributes(treelstm.models_dir) == nil then
  lfs.mkdir(treelstm.models_dir)
end

-- get paths
local file_idx = 1
local subtask = fine_grained and '5class' or '2class'
local predictions_save_path, model_save_path
while true do
  predictions_save_path = string.format(
    treelstm.predictions_dir .. '/sent-%s.%s.%dl.%dd.%d.pred', args.model, subtask, args.layers, args.dim, file_idx)
  model_save_path = string.format(
    treelstm.models_dir .. '/sent-%s.%s.%dl.%dd.%d.th', args.model, subtask, args.layers, args.dim, file_idx)
  if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
for i = 1, test_predictions:size(1) do
  predictions_file:writeInt(test_predictions[i])
end
predictions_file:close()
]]--

-- write results to disk
if lfs.attributes(treelstm.results) == nil then
  lfs.mkdir(treelstm.results)
end
local subtask = fine_grained and '5class' or '2class'
local results_save_path = string.format(treelstm.results .. '/sent_' .. sys.clock() .. '.out')
print('writing results to ' .. results_save_path)
best_dev_model:save_results(results_save_path)

-- write model to disk
--print('writing model to ' .. model_save_path)
--best_dev_model:save(model_save_path)

-- to load a saved model
-- local loaded = model_class.load(model_save_path)
