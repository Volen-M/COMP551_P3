import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Net import *
from training import train
from torch.utils.data.sampler import SubsetRandomSampler

#Normalize our image pixels to be in [-1,1]. image=(image-mean)/std
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Loading the data sets from the Library
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)
#The classes our model will attempt to predict
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######## TESTING ########
PATH = './cifar_net.pth'

##### SET TEST ######
def test(net,  test_set):
    #Load the trained neural network that was saved at PATH
    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        # For each batch of size 4
        for data in test_set:
            images, labels = data
            # get predicted output of CNN
            outputs = net(images)
            # For each of 4 instances, get the class with highest power
            _, predicted = torch.max(outputs.data, 1)
            # Add 4 every iteration of the forloop. At the end total holds the number of test points that were predicted!
            total += labels.size(0)
            # anding the vectors (predicted and labels), then summing the values inside the vector. Adding all of it to the variable correct.
            # This variable holds the total number of instances who's label was correctly predicted!
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return 100 * correct / total

def test_train(net, train_set):
    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    #Impacts the autograd engine adn deactivates it => reduces memory usage and speed up (https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
    with torch.no_grad():
        # For each batch of size 4
        for data in train_set:
            images, labels = data
            # get predicted output of CNN
            outputs = net(images)
            # For each of 4 instances, get the class with highest power
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # anding the vectors (predicted and labels), then summing the values inside the vector. Adding all of it to the variable correct.
            # This variable holds the total number of instances who's label was correctly predicted!
            correct += (predicted == labels).sum().item()
    print('training Accuracy of the network on the 50000 train images: %d %%' % (
        100 * correct / total))
    return 100 * correct / total

#### Class Performance ####
def class_perf(net):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



def cross_validation(lr):
    acc = 0
    #5-cross validation
    for k in range(0,5):

        # load the dataset
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True,
            download=True, transform=transform,
        )

        valid_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True,
            download=True, transform=transform,
        )

        net = Net()
        num_train = len(train_dataset)
        indices = list(range(num_train))
        #We want to split the training set in 5 parts
        split = int(np.floor(1/5 * num_train))

        #We randomly select 40000 instances of the training to train on and 10000 to validate on
        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_set = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, sampler=train_sampler,
            num_workers=0,
        )
        valid_set = torch.utils.data.DataLoader(
            valid_dataset, batch_size=4, sampler=valid_sampler,
            num_workers=0,
        )

        #We train the model with 6 epochs as 6 epochs was the peak test accuracy in Exp1
        #And also appeared to be the elbow for the loss function graph (Exp1.0)
        train(6, net, train_set, lr)
        acc += test(net, valid_set)
    print("The accuracy score for 5-cross validation is: ", acc/5, "%\n")
    return acc/5

#### Test Perf as function of training Epochs #####
def test_1():
    net = Net()
    train(0, net, trainloader, 0.001)
    test_train(net, trainloader)
    test(net, testloader)
    for i in range(0,15):
        train(2, net, trainloader, 0.001)
        test_train(net, trainloader)
        test(net, testloader)

#Additional experiments!(For EXP2.0, EXP2.1, I reused test_1() but with Leaky ReLU)
def test_2():
    #Gonna change the activation function and cross validate and then reuse the activation with the best validation.
    # I was not able to implement it in one function, so I am manually changing the activation in net and then cross validating
    x1 = cross_validation(0.001)
    x2 = cross_validation(0.01)
    x3 = cross_validation(0.1)
    print(x1, x2, x3)

test_1()
#test_2()
#TODO: Hyper parameter Tuning (#number of layers, try 1-4/5 CNN, 1-4/5 fc (epochs 1,5,10,15))