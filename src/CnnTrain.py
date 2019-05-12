import os

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as utils

import matplotlib.pyplot as plt

from src.Train import trainFeatures, trainLabels
from src.util.MapStrings import map_strings_to_int

# Constant
IMAGE_SIZE = 300

# --- HYPER-PARAMETERS --
# Convolution
params = {
    'featureMaps': [1, 32, 25],
    'numOfConvLayers': 1,
    'convKernelSizes': [5],
    'convStrides': [2],
    'convPaddings': [0],
    'poolStrides': [2],
    'poolKernelSizes': [2],
    'numOfFullyConnectedLayers': 2,
    'hiddenLayers': [2500]
}


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

# Neural Network
num_epochs = 10
num_classes = 25
batch_size = 100
learning_rate = 0.001

# Storing model path
MODEL_STORE_PATH = '../generatedModels/'

# --- DATA TRANSFORM AND LOADING ---

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

mapped_trainLabels = map_strings_to_int(trainLabels)

trainFeatures_tensor = torch.from_numpy(np.asarray(trainFeatures,dtype='float32'))
trainLabels_tensor = torch.from_numpy(mapped_trainLabels)

# ***** Data loader *****
# ____TRAIN LOADER____
train_dataset = utils.TensorDataset(trainFeatures_tensor,trainLabels_tensor) # create your datset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Test Features and labels
testFeatures = []
testLabels = []
for imageName in os.listdir("../samples/test"):
    imageLabel = imageName[0]
    testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
    testLabels.append(imageLabel)

# Converting the Test features/labels from Numpy to Tensors
testFeatures_tensor = torch.from_numpy(np.asarray(testFeatures,dtype='float32'))
mapped_testLabels = map_strings_to_int(testLabels)
testLabels_tensor = torch.from_numpy(mapped_testLabels)

# ____TEST LOADER____
test_dataset = utils.TensorDataset(testFeatures_tensor,testLabels_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- END OF DATA LOADING AND TRANSFORM --

# Convolutional Neural Network


class ConvNet(nn.Module):
    def __init__(self, params):
        super(ConvNet, self).__init__()

        # Initially Image Size (300)
        out_size = IMAGE_SIZE

        # Convolutional layers
        for i in range(params['numOfConvLayers']):
            # Parameters
            conv_in = params['featureMaps'][i]
            conv_out = params['featureMaps'][i + 1]

            # Convolution
            kernel_size = params['convKernelSizes'][i]
            stride = params['convStrides'][i]
            padding = params['convPaddings'][i]

            # Output size after convolution
            out_size = outputSize(out_size, kernel_size, stride, padding)

            # Pooling
            p_kernel_size = params['poolKernelSizes'][i]
            p_stride = params['poolStrides'][i]

            # Output size after pooling
            out_size = outputSize(out_size, p_kernel_size, p_stride, 0)

            layer = """nn.Sequential(nn.Conv2d(%d, %d, kernel_size=%d, stride=%d, padding=%d),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=%d, stride=%d))""" % (conv_in, conv_out, kernel_size, stride, padding, p_kernel_size, p_stride)

            exec("self.layer%d = %s" % (i + 1, layer))

        # Fully connected
        final_feature_maps = params['featureMaps'][-2]
        num_labels = params['featureMaps'][-1]

        for i in range(params['numOfFullyConnectedLayers']):
            if i == 0:
                fc = """nn.Linear(%d, %d)""" % (out_size * out_size * final_feature_maps, params['hiddenLayers'][i])
                exec("self.fc%d = %s" % (i + 1, fc))
            elif i == params['numOfFullyConnectedLayers'] - 1:
                fc = """nn.Linear(%d, %d)""" % (params['hiddenLayers'][i - 1], num_labels)
                exec("self.fc%d = %s" % (i + 1, fc))
            else:
                fc = """nn.Linear(%d, %d)""" % (params['hiddenLayers'][i], params['hiddenLayers'][i + 1])
                exec("self.fc%d = %s" % (i + 1, fc))

    def forward(self, x):

        out = x

        out = self.layer1(out)
        # out = self.layer2(out)

        # for i in range(params['numOfConvLayers']):
        #     print("out = self.layer%d(out)" % (i + 1))
        #     out = exec("self.layer%d(out)" % (i + 1))

        out = out.reshape(out.size(0), -1)

        # for i in range(params['numOfFullyConnectedLayers']):
        #     print("out = self.fc%d(out)" % (i + 1))
        #     out = exec("self.fc%d(out)" % (i + 1))

        out = self.fc1(out)
        out = self.fc2(out)

        return out


# CNN Model
model = ConvNet(params)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Run the forward pass
        images = images.reshape(100, 1, 300, 300)
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Normalization
        loss_list.append(loss.item() / 1000)

        # Back-propagation and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs, 1)

        correct = 0
        for j in range(0, 100):
            if predicted[j] == labels[j]:
                correct +=1
        acc_list.append(correct / total)

        # Log
        print("Iteration:", i)
        print("Accuracy:",(correct/total))
        print("Loss:", loss.item() / 1000)
        print("--------")

        if (i + 1) % 17 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    numOfSamples = 100
    for images, labels in test_loader:
        if images.shape[0] < 100:
            break

        images = images.reshape(numOfSamples, 1, 300, 300)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct = 0
        for j in range(0, 100):
            if predicted[j] == labels[j]:
                correct += 1

    print('Test Accuracy of the model on the 400 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

# Train Loss vs Iterations
del loss_list[0]
