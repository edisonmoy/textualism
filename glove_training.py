from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os as os
from sklearn.model_selection import train_test_split

class GloveDataset:
	def __init__(self, text, n_words=200000, window_size=5):
		    self._window_size = window_size
		    self._tokens = text.split(" ")[:n_words]
		    word_counter = Counter()
		    word_counter.update(self._tokens)
		    #Might do some changes here to reduce the Size of the vocabulary
		    self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
		    self._id2word = {i:w for w, i in self._word2id.items()}
		    self._vocab_len = len(self._word2id)

		    self._id_tokens = [self._word2id[w] for w in self._tokens]

		    self._create_coocurrence_matrix()

		    print("\t# of words: {}".format(len(self._tokens)))
		    print("\tVocabulary length: {}".format(self._vocab_len))

	def _create_coocurrence_matrix(self):
		cooc_mat = defaultdict(Counter)
		for i, w in enumerate(self._id_tokens):
			start_i = max(i - self._window_size, 0)
			end_i = min(i + self._window_size + 1, len(self._id_tokens))
			for j in range(start_i, end_i):
				if i != j:
					c = self._id_tokens[j]
					cooc_mat[w][c]=1.

		for i, w in enumerate(self._id_tokens):
			start_i = max(i - self._window_size, 0)
			end_i = min(i + self._window_size + 1, len(self._id_tokens))
			for j in range(start_i, end_i):
				if i != j:
					c = self._id_tokens[j]
					cooc_mat[w][c] += 1 / abs(j-i)
			self._i_idx=list()
			self._j_idx=list()
			self._xij=list()

		#Create indexes and x values tensors
		for w, cnt in cooc_mat.items():
			for c, v in cnt.items():
				self._i_idx.append(w)
				self._j_idx.append(c)
				self._xij.append(v)

		self._i_idx = torch.LongTensor(self._i_idx)
		self._j_idx = torch.LongTensor(self._j_idx)
		self._xij = torch.FloatTensor(self._xij)

	def get_batches(self, batch_size):
		#Generate random idx
		rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

		for p in range(0, len(rand_ids), batch_size):
		    batch_ids = rand_ids[p:p+batch_size]
		    yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


def get_batches(self, batch_size):
    #Generate random idx
    rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

    for p in range(0, len(rand_ids), batch_size):
        batch_ids = rand_ids[p:p+batch_size]
        yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x

def weight_func(x, x_max, alpha):
	wx = (x/x_max)**alpha
	wx = torch.min(wx, torch.ones_like(wx))
	return wx

def wmse_loss(weights, inputs, targets):
	loss = weights * F.mse_loss(inputs, targets, reduction='none')
	return torch.mean(loss)

def reconstruct_sentence(meta_arr):
	#Reconstruct sentence as string from array of words
	clean_sentence_helper_arr=[]
	for list_words in meta_arr:
		help_sentence=' '.join(word for word in list_words)
		clean_sentence_helper_arr.append(help_sentence)
	return clean_sentence_helper_arr

def reconstruct_text(sentence_arr):
	helper_string=""
	for index,sentence in enumerate(sentence_arr):
		helper_string+=sentence+" "
	return helper_string


# Given hyper parameters and year, return loss values
def train_model(year, epochs, batch_size, x_max, embed_dim, num_words, lr_val, alpha):
  losses = []
  all_validation_losses =[]

  # Create new model folder
  model_path="./models/Data_Justia_"+str(year)+"/"
  try:
	  os.mkdir(model_path)
  except OSError:
	  print ("Creation of the directory %s failed" % model_path)


  # Load and split data into train/validation sets
  print("Loading data from "+ str(year))
  data_path = "./cleaned_data/"
  sen_arr=np.load(data_path + "Sentence_Array_Justia_"+str(year)+".npy", allow_pickle=True)
  sen_arr_train, sen_arr_validation = train_test_split(sen_arr,test_size=validation_set_size)

  #Process training data
  print("\nProcessing training data... ")
  sentence_arr_train = reconstruct_sentence(sen_arr_train)
  sen_text_train = reconstruct_text(sentence_arr_train)
  dataset_train = GloveDataset(sen_text_train,num_words)
  # np.save("Words_dict",dataset_train._id2word)
  n_batches_train = int(len(dataset_train._xij) / batch_size)
  glove_train = GloveModel(dataset_train._vocab_len, embed_dim)
  optimizer_train = optim.Adagrad(glove_train.parameters(), lr=lr_val)

  #Process validation data
  print("\nProcessing validation data... ")
  sentence_arr_validation = reconstruct_sentence(sen_arr_validation)
  sen_text_validation = reconstruct_text(sentence_arr_validation)
  dataset_validation = GloveDataset(sen_text_validation,num_words)
  n_batches_validation = int(len(dataset_validation._xij) / batch_size)

  # np.save("Words_dict",dataset_validation._id2word)


  print ("\nStart Training")
  loss_list = []
  validation_loss = []
  iteration_i = 0
  for e in range(1, epochs+1):
    loss_per_batch = []
    validation_loss_per_epoch = []
    batch_i = 0
    for x_ij, i_idx, j_idx in dataset_train.get_batches(batch_size):
      batch_i += 1
      iteration_i += 1
      optimizer_train.zero_grad()
      outputs = glove_train(i_idx, j_idx)
      weights_x = weight_func(x_ij, x_max, alpha)
      loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
      loss_list.append(loss.item())
      np.save(model_path+"Losses_test_Embed_Dim="+str(embed_dim)+"_lr="+str(lr_val),np.array(loss_list))
      loss.backward()
      optimizer_train.step()
      if batch_i % 100 == 0:
        loss_per_batch.append((iteration_i, np.mean(loss_list[-20:])))
        print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, epochs, batch_i, n_batches_train, np.mean(loss_list[-20:])))

      # Run validation set after each 200th iteration
      if batch_i % 200 == 0:
        validation_loss_temp = []
        for x_ij, i_idx, j_idx in dataset_validation.get_batches(batch_size):
          outputs = glove_train(i_idx, j_idx)
          weights_x = weight_func(x_ij, x_max, alpha)
          loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
          validation_loss_temp.append(loss.item())
        validation_loss_per_epoch.append((iteration_i, np.mean(validation_loss_temp[-20:])))
        print("Epoch: {}/{} \t Validation \t \t Loss: {}".format(e, epochs, np.mean(validation_loss_temp[-20:])))

    # End of epoch
    print("---------------------------------------------")
    #Save Weights after each epoch
    print("Saving model weights...")
    np.save(model_path+"Weight_Matrix1_Embed_Dimension_"+str(EMBED_DIM)+"_lr="+str(lr_helper), glove_train.wi.weight.detach().numpy())
    np.save(model_path+"Weight_Matrix2_Embed_Dimension_"+str(EMBED_DIM)+"_lr="+str(lr_helper), glove_train.wj.weight.detach().numpy())
    np.save(model_path+"ID2WORD_Embed_EMBED_DIM="+str(EMBED_DIM)+"_lr="+str(lr_helper),dataset_train._id2word)
    np.save(model_path+"WORD2ID_Embed_EMBED_DIM="+str(EMBED_DIM)+"_lr="+str(lr_helper),dataset_train._word2id)
    torch.save(glove_train.state_dict(), "text8.pt")


    # Update losses
    all_validation_losses.append(validation_loss_per_epoch)
    losses.append(loss_per_batch)
  return losses , all_validation_losses



N_EPOCHS 	= 10
BATCH_SIZE 	= 1000
X_MAX 		= 10
EMBED_DIM	= 200
number_words= 1000000
lr_helper	= 0.1
validation_set_size = 0.2

#Adapted From GlovePaper
ALPHA 		= 0.75

# Train models
for x in range(1962, 2000):
	train_model(x, N_EPOCHS, BATCH_SIZE, X_MAX, EMBED_DIM, number_words, lr_helper, ALPHA)
	print("\n \n ----------------------------------------------------------------- \n\n")

for x in range(2015, 2020):
	train_model(x, N_EPOCHS, BATCH_SIZE, X_MAX, EMBED_DIM, number_words, lr_helper, ALPHA)
	print("\n \n ----------------------------------------------------------------- \n\n")
