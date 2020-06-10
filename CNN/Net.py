import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Initial images are of size 3x32x32
######## INITIALIZING THE NEURAL NET #########
class Net(nn.Module):
    #A lot of hyper parameter tuning can happen here!
    def __init__(self):
        super(Net, self).__init__()

        #EXP_1
        # creates a convolutional layer with 3 inputs, 6 filters and filter_size 5.
        self.conv1 = nn.Conv2d(3, 6, 5)
        # max pool layer kernel size 2 with stride 2. the height & width of conv1/conv2 OUT will be decreased by factor of 2.
        self.pool = nn.MaxPool2d(2, 2)
        # creates a convolutional layer with 6 input, 16 filters and filter_size 5.
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layer with size 16 (conv2_Out) * 5 * 5 where 5 x 5 is the size of each output image and 120 out_features.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #We are expecting input matrices of size 5x5 => we had conv1, pool, conv2, pool then fc1. because (32-5)/1 + 1 = 28, 28/2 = 14, (14-5)/1 + 1 = 10, 10/2 = 5.
        # fully connected layer with input 120 and output 84
        self.fc2 = nn.Linear(120, 84)
        # fully connected layer with input 84 and output 10. FINAL LAYER
        self.fc3 = nn.Linear(84, 10)

        #EXP_3: Changing the network

    # calculates the energies per class (one iteration of the neural network). Instance passes through the neurons and activations through the later to the end
    def forward(self, x):
        #Test_1
        # This confirms what I described above, we must be pooling twice to get input matrices of size 5x5 in fc1
        # maxpool conv1_OUT after applying ReLU
        x = self.pool(F.relu(self.conv1(x)))
        # maxpool conv2_OUT after applying ReLU
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        # ReLU to fc1
        x = F.relu(self.fc1(x))
        # ReLU to fc2
        x = F.relu(self.fc2(x))
        # x holds the energies of each of 10 classes
        x = self.fc3(x)                         #Final output is one of 10 different classes (airplane, automobile, bird, cat, deer ,dog, frog, horse, ship, truck)

        #Test_2.0 sigmoid
        #x = self.pool(torch.sigmoid(self.conv1(x)))
        #x = self.pool(torch.sigmoid(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = torch.sigmoid(self.fc1(x))
        #x = torch.sigmoid(self.fc2(x))
        #x = self.fc3(x)

        #Test 2.1 leaky reLu
        #x = self.pool(F.leaky_relu(self.conv1(x)))
        #x = self.pool(F.leaky_relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc2(x))
        #x = self.fc3(x)

        #Test 2.2 softplus
        #x = self.pool(F.softplus(self.conv1(x)))
        #x = self.pool(F.softplus(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        #x = F.softplus(self.fc1(x))
        #x = F.softplus(self.fc2(x))
        #x = self.fc3(x)
        return x