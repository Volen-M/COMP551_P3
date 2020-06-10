import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from Net import *
from train import *

import numpy as np
import matplotlib.pyplot as plt


## Validation  test taken from https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
def validate(loss_arr, accuracy_arr, model, testLoader, cuda=None):
	model.eval()
	val_loss, correct = 0, 0
	with torch.no_grad():
		for data, target in testLoader:
			if cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			output = model(data)
			val_loss += torch.nn.functional.nll_loss(output, target).data
			pred = output.data.max(1)[1] # get the index of the max log-probability
			correct += pred.eq(target.data).cpu().sum()

		val_loss /= len(testLoader)
		loss_arr.append(val_loss)

		accuracy = 100. * correct / len(testLoader.dataset)
		accuracy_arr.append(accuracy)
		
def main():

		
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
											download=True, transform=transform)
	trainLoader = torch.utils.data.DataLoader(trainset, batch_size=4,
											  shuffle=True, num_workers=0)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										   download=True, transform=transform)
	testLoader = torch.utils.data.DataLoader(testset, batch_size=4,
											 shuffle=False, num_workers=0)

	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


	learning_rates =  [0.001, 0.01, 0.1]

	
		
	learning_rates =  [0.001]	
	for i in range(len(learning_rates)):
		loss_arr = []
		accuracy_arr = []
		cuda = torch.cuda.is_available()
		epochs = 20
		model = Net(1, 100, "relu")
		if cuda:
			model.cuda()
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[i], momentum=0.5)
	
		for epoch in range(1,epochs + 1):
			train(epoch, model, trainLoader,optimizer, cuda)
			validate(loss_arr, accuracy_arr, model, testLoader, cuda=cuda)
		
		print(accuracy_arr)
		print(loss_arr)
		epochs = [i for i in range(1,epochs + 1)]
		print(epochs)
	
		
	
	loss_arr = []
	accuracy_arr = []
	
	learning_rates =  [0.001, 0.01, 0.1]	
	for i in range(len(learning_rates)):
		loss_arr = []
		accuracy_arr = []
		cuda = torch.cuda.is_available()
		epochs = 6
		model = Net(1, 100, "relu")
		if cuda:
			model.cuda()
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[i], momentum=0.5)
	
		for epoch in range(1,epochs + 1):
			train(epoch, model, trainLoader,optimizer, cuda)
			validate(loss_arr, accuracy_arr, model, testLoader, cuda=cuda)
		
		print(accuracy_arr)
		print(loss_arr)
		epochs = [i for i in range(1,epochs + 1)]
		print(epochs)
	
		
	layers =  [1, 2]
	for i in range(len(layers)):
		loss_arr = []
		accuracy_arr = []
		cuda = torch.cuda.is_available()
		epochs = 6
		model = Net(layer[i], 100, "relu")
		if cuda:
			model.cuda()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
	
		for epoch in range(1,epochs + 1):
			train(epoch, model, trainLoader,optimizer, cuda)
			validate(loss_arr, accuracy_arr, model, testLoader, cuda=cuda)
		
		print(accuracy_arr)
		print(loss_arr)
		epochs = [i for i in range(1,epochs + 1)]
		print(epochs)
	
	units_per_layer = [50,60,70,80,100]
	
	for i in range(len(units_per_layer)):
		loss_arr = []
		accuracy_arr = []
		cuda = torch.cuda.is_available()
		epochs = 5
		model = Net(1, units_per_layer[i], "relu")
		if cuda:
			model.cuda()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
	
		for epoch in range(1,epochs + 1):
			train(epoch, model, trainLoader,optimizer, cuda)
			validate(loss_arr, accuracy_arr, model, testLoader, cuda=cuda)
		
		print(accuracy_arr)
		print(loss_arr)
		epochs = [i for i in range(1,epochs + 1)]
		plt.plot(epochs, accuracy_arr, color='b')
		plt.scatter(epochs, accuracy_arr)
		plt.title("Accuracy of Relu MLP with " + units_per_layer[i] + " Units per layer")
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy with Relu")
		plt.show()
		
		plt.plot(epochs, loss_arr)
		plt.title("Loss of Relu MLP with " + units_per_layer[i] + " Units per layer")
		plt.xlabel("Epochs")
		plt.ylabel("Loss with Relu")
		plt.show()

	loss_arr = []
	accuracy_arr = []
	learning_rates =  [0.001, 0.01, 0.1]	
	for i in range(len(learning_rates)):
		cuda = torch.cuda.is_available()
		epochs = 20
		model = Net(1, 100, "sigmoid")
		if cuda:
			model.cuda()
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[i], momentum=0.5)
	
		for epoch in range(1,epochs + 1):
			train(epoch, model, trainLoader,optimizer, cuda)
			validate(loss_arr, accuracy_arr, model, testLoader, cuda=cuda)
		
		epochs = [i for i in range(1,epochs + 1)]
		print(accuracy_arr)
		print(loss_arr)
		print(epochs)
	
main()