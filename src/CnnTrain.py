import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

'''

'''
CONST_INPUT_CHANNEL = 1
output_channel = [5,2]
kernel_size = [3,4]
stride = [2,1]
padding = [1,1]
num_of_conv_layer = 2

pool_kernel_size = [2,2]
pool_stride = [2,1]
pool_padding = [1,2]

def initialize_params(output_channel,kernel_size,stride,padding,num_of_conv_layer, pool_kernel_size,pool_stride, pool_padding):
    params = []
    for i in range(num_of_conv_layer):
        params.append({
            'output_channel': output_channel[i],
            'kernel_size': kernel_size[i],
            'stride': stride[i],
            'padding': padding[i],
            'num_of_conv': num_of_conv_layer[i],
            'pool_kernel_size': pool_kernel_size[i],
            'pool_stride' : pool_stride[i],
            'pool_padding' : pool_padding[i]
    })
    return params

params = initialize_params(output_channel,kernel_size,stride,padding,num_of_conv_layer,pool_kernel_size,pool_stride, pool_padding)

class CNN(torch.nn.Module):

    '''
        Constructing CNN
    '''

    def __init__(self, params):
        super(CNN, self).__init__()
        self.conv_outputs = []
        for i in range(params['num_of_conv_layer']):
            self.conv_outputs.append( torch.nn.Conv2d(CONST_INPUT_CHANNEL, params['output_channel'][i]
                                                      ,kernel_size=params['kernel_size'][i],
                                                      stride= params['stride'][i], padding=params['padding'][i]))
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        # 64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 * 16)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return (x)


def train_net(trainFeatures,trainLabels):
    return 0