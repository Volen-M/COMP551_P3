import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as T
from torch.autograd import Variable
import torch.optim as optim
from Net import *



######## LOSS FUNCTION + OPTIMIZER #########
######### TRAINING ########
def train(epoch, model,trainLoader, optimizer, cuda = None, interval=2000):

	model.train()
	
	for i, (data, target) in enumerate(trainLoader):
	
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad() 
		output = model(data)
		loss = T.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		
		