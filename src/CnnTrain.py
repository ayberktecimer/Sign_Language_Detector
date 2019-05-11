import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
import torch.utils.data as utils
import os
import cv2

# Hyperparameters
num_epochs = 6
num_classes = 25
batch_size = 100
learning_rate = 0.001

DATA_PATH = '../MNISTData'
MODEL_STORE_PATH = '../pytorch_models'

# transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
#train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
#test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)
from src.Train import trainFeatures, trainLabels


def map_strings_to_int(trainLabels):
    mapped_trainLabels = []
    dict = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'F': 5,
        'G': 6,
        'H': 7,
        'I': 8,
        'K': 9,
        'L': 10,
        'M': 11,
        'N': 12,
        'O': 13,
        'P': 14,
        'Q': 15,
        'R': 16,
        'S': 17,
        'T': 18,
        'U': 19,
        'V': 20,
        'W': 21,
        'X': 22,
        'Y': 23,
        'Z': 24,
    }
    for i in trainLabels:
        mapped_trainLabels.append(dict[i])
    return np.asarray(mapped_trainLabels,dtype='int64')
mapped_trainLabels = map_strings_to_int(trainLabels)

trainFeatures_tensor = torch.from_numpy(np.asarray(trainFeatures,dtype='float32'))
trainLabels_tensor = torch.from_numpy(mapped_trainLabels)

# Data loader
train_dataset = utils.TensorDataset(trainFeatures_tensor,trainLabels_tensor) # create your datset
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

testFeatures = []
testLabels = []
for imageName in os.listdir("../samples/test"):
    imageLabel = imageName[0]
    testFeatures.append(cv2.imread("../samples/test/" + imageName, 0).ravel())
    testLabels.append(imageLabel)


testFeatures_tensor = torch.from_numpy(np.asarray(testFeatures,dtype='float32'))
mapped_testLabels = map_strings_to_int(testLabels)
testLabels_tensor = torch.from_numpy(mapped_testLabels)

test_dataset = utils.TensorDataset(testFeatures_tensor,testLabels_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Convolutional neural network (two convolutional layers)
# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(75 * 75 * 64, 2500)
        self.fc2 = nn.Linear(2500, 25)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()
#model = model.float()
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
        images = images.reshape(100,1,300,300)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct = 0
        for i in range(0,100):
            if predicted[i] == labels[i]:
                correct +=1
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
        print("")
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
show(p)

def train_net(trainFeatures,trainLabels):
    return 0