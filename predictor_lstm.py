import numpy as np
import torch 
import torch.nn as nn
from nltk.tokenize import TreebankWordTokenizer
from utils import preprocess
import json
import os
import sys


class SentimentClassifier(nn.Module):

	def __init__(self, n_vocab, embedding_dim, n_hidden, n_layers, n_classes, embedding_mat, dropout_rate=0.3):

		super(SentimentClassifier, self).__init__()
		self.V = n_vocab
		self.D = embedding_dim
		self.H = n_hidden
		self.L = n_layers
		self.O = n_classes

		self.embedding = nn.Embedding(self.V, self.D)
		self.embedding.weight = nn.Parameter(embedding_mat, requires_grad=False)
		self.lstm = nn.LSTM(self.D, self.H, self.L, dropout=dropout_rate, batch_first=True)
		
		self.fc = nn.Sequential(nn.ReLU(),
								nn.Linear(self.H*2, 64),
								nn.ReLU(),
								nn.Dropout(dropout_rate),
								nn.Linear(64, 16),
								nn.ReLU(),
								nn.Dropout(dropout_rate),
								nn.Linear(16, self.O),
								nn.Sigmoid())


	def forward(self, X):
		h0, c0 = torch.zeros(self.L, X.size(0), self.H), torch.zeros(self.L, X.size(0), self.H)
		out = self.embedding(X)
		out, (h,c) = self.lstm(out, (h0,c0))
		avg_pool = torch.mean(out, 1)
		max_pool, _ = torch.max(out, 1)
		conc = torch.cat((avg_pool, max_pool), 1)
		out = self.fc(conc)

		return out


def predict(model, inputs):

	model.eval()
	inputs = inputs.view(1, -1)
	print(inputs)
	print(inputs.size())
	outputs = model(inputs)
	outputs = np.round(outputs.detach().numpy())
	outputs = outputs.reshape(-1,)

	if outputs[0] == 0.:
		return "negative"
	else:
		return "positive"



def main(data):
	preprocessed_data = preprocess(data, lemmatize=True)[0]

	if not os.path.exists("./MODELS/lstm_model.pt"):
		os.system("gdown https://drive.google.com/uc?id=1daQttFvSBiJUckTOT3nH7L_5yYe7VcQl")
		os.system("mv lstm_model.pt MODELS/")


	if not os.path.exists("./MODELS/features.npz"):
		os.system("gdown https://drive.google.com/uc?id=1D6tgepG4ArWCFXe40FHtcQLdR2-sjFh1")
		os.system("mv features.npz MODELS/")



	encoded_data = list()
	tokenizer = TreebankWordTokenizer()
	tokens = tokenizer.tokenize(preprocessed_data)

	f = open("./MODELS/vocabulary.json", "r")
	vocab = json.load(f)
	f.close()

	print("Tokens ", tokens)
	for t in tokens:
		if t in vocab:
			encoded_data.append(vocab[t])

		else:
			encoded_data.append(0)


	embedding_mat = np.load("./MODELS/features.npz")
	embedding_mat = torch.from_numpy(embedding_mat.astype(np.float32))

	encoded_data = torch.from_numpy(np.array(encoded_data)).long()

	model = SentimentClassifier(len(vocab), 300, 128, 2, 1, embedding_mat)
	model.load_state_dict(torch.load("./MODELS/lstm_model.pt", map_location=torch.device("cpu"))["model"])

	review_sentiment = predict(model, encoded_data)

	return review_sentiment
	


	








