import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as T
import torch.optim as optim


#Initial images are of size 3x32x32
######## INITIALIZING THE NEURAL NET #########
class Net(nn.Module):
	def __init__(self,  nb_hidden_layers, units_per_layer, activation_func,  dropout_rate = 0.5):
		super(Net, self).__init__()
		self.units_per_layer = units_per_layer
		self.activation_func = activation_func
		self.nb_hidden_layers = nb_hidden_layers
		
		self.fc1 = nn.Linear(32*32*3, units_per_layer)
		self.fc1_drop = nn.Dropout(dropout_rate)
		
		if nb_hidden_layers > 1:
			self.fc2 = nn.Linear(units_per_layer, units_per_layer)
			self.fc2_drop = nn.Dropout(dropout_rate)
		if nb_hidden_layers == 3:
			self.fc3 = nn.Linear(units_per_layer, units_per_layer)
			self.fc3_drop = nn.Dropout(dropout_rate)
		self.out = nn.Linear(units_per_layer, 10)

	def forward(self, x):
		x = x.view(-1, 32*32*3)
		if self.activation_func == "relu":
			x = T.relu(self.fc1(x))
		elif self.activation_func == "sigmoid":
			x = torch.sigmoid(self.fc1(x))
		x = self.fc1_drop(x)
		if self.nb_hidden_layers >= 2:
			if self.activation_func == "relu":
				x = T.relu(self.fc2(x))
			elif self.activation_func == "sigmoid":
				x = torch.sigmoid(self.fc2(x))
			x = self.fc2_drop(x)
		if self.nb_hidden_layers == 3:
			if self.activation_func == "relu":
				x = T.relu(self.fc3(x))
			elif self.activation_func == "sigmoid":
				x = torch.sigmoid(self.fc3(x))
			x = self.fc3_drop(x)
		
		return T.log_softmax(self.out(x))
