import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Net import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######## LOSS FUNCTION + OPTIMIZER #########
######### TRAINING ########
def train(epochs, net, train_set, lr):
    #CrossEntropyLoss assuming a Bernoulli model (same loss function as with logistic regression)
    criterion = nn.CrossEntropyLoss()
    # Stochastic gradient descent with learning rate lr (default: 0.001), and momentum (beta) value 0.9
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    #Number of epochs = number of iterations through the dataset
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients. x.grad=0 for each parameter to optimize
            optimizer.zero_grad()

            # forward + backward + optimize
            # calculate the predicted labels from the current net
            outputs = net(inputs)
            # calculate the loss for the predicted labels vs actual labels
            loss = criterion(outputs, labels)
            # Optimize the parameters.
            # computes dloss/dx for every parameter x which has requires_grad=True. They are stored into x.grad
            loss.backward()
            # Apply the change to the parameters => these parameters are the same instances from the net, so the parameters of the net are changed here also.
            #updates the values of x using teh gradient x.grad (x += -lr*x.grad + 0.9 x)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    #Saves the trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)