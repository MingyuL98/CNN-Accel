import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import random_split
import torch.nn.functional as F
from torchsummary import summary    # for model summary

from ConvNet import MyConvNet

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

# argument parser
parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - CNN example')
parser.add_argument('--batch-size', type=int, default=100, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--enable-cuda', type=bool, default=1, help='Enable traning on gpu')
parser.add_argument('--activation-func', type=str, default='relu', help='Activation function [relu, sigmoid, tanh, none]')
parser.add_argument('--train-step', type=int, default=0, help='-1: pruning, 0: regular, 1: step-1, 2: fine tuning')
parser.add_argument('--elw-thr', type=float, default=0.3, help='when train-step==-1, zero out weights lower then threshold')
parser.add_argument('--elw-frac', type=float, default=0.3, help='when train-step==-1, prune weights of lowest frac')

args = parser.parse_args()

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr


# Model constructions
seed = 0
activation = args.activation_func
train_step = args.train_step



# Paths
PATH_STEP1 = "models/model_step1.pth"
PATH_STEP2 = "models/model_step2.pth"

# pruning
elw_frac = args.elw_frac
elw_thr = args.elw_thr

## enable training on gpu
enable_cuda = args.enable_cuda

# Device settings (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
print('Train using', device)

# set random seed 
if(train_step > 0):
    torch.manual_seed(seed)

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root ='./data',
        train = True,
        transform = transforms.ToTensor(),
        download = True)

test_dataset = dsets.MNIST(root ='./data',
        train = False,
        transform = transforms.ToTensor())

'''
# split trainset
if(train_step > 0):
    train_dataset_1, train_dataset_2 = random_split(train_dataset, [len(train_dataset)//2, len(train_dataset) - len(train_dataset)//2])
    train_dataset = train_dataset_2 if train_step > 2 else train_dataset_1
'''

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False)

model = MyConvNet()
# summary(model, (1, 28, 28))

# load model (train_step = 2 or -1)        
if (train_step == 2):
    model.load_state_dict(torch.load(PATH_STEP1))
elif (train_step < 0):
    model.load_state_dict(torch.load(PATH_STEP2))

model = model.to(device)

# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

# Training the Model
training_loss = []
training_acc = []
testing_loss = []
testing_acc = []

step_total = len(train_dataset) // batch_size

for epoch in range(num_epochs):
    loss_each_epoch = 0.0
    acc_each_epoch = 0.0
    correct = 0.0
    total = 0.0
   
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # calculate the loss of each epoch
        loss_each_epoch += loss.data.item()
        
        # calculate the accuracy of each epoch
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        total += labels.size(0)

        if (i + 1) % 100 == 0:
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, step_total, loss.data.item()))
    
    acc_each_epoch = 100 * float(correct)/total
    training_loss.append(loss_each_epoch/len(train_loader))
    training_acc.append(acc_each_epoch)

    # Test the Model (show how test acc changes throughout the training)
    with torch.no_grad():
        correct = 0.0
        total = 0.0

        for images, labels in test_loader:
            images = Variable(images).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)

            correct += (predicted == labels).sum()

        
        acc_each_epoch = 100 * float(correct)/total

        print('Accuracy of the model on the 10000 test images: % .2f %%' % acc_each_epoch)
        print('Loss of the model on the 10000 test images: %.4f'  % (loss))
    
        testing_acc.append(acc_each_epoch)
        testing_loss.append(loss)


if (train_step == 1):
    torch.save(model.state_dict(), PATH_STEP1)
elif (train_step == 2):
    torch.save(model.state_dict(), PATH_STEP2)


# trainig loss/accuracy VS testing loss/accuracy
figname = str(batch_size) + '-' + str(learning_rate) + '.png'

plt.plot(training_loss, label='Training loss')
plt.plot(testing_loss, label='Testing loss')
plt.xticks(np.arange(0, num_epochs, step=1), np.arange(1, num_epochs+1, step=1).astype('str').tolist())
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.savefig('figs/'+'Loss'+figname)
plt.show()

plt.plot(training_acc, label='Training accuracy')
plt.plot(testing_acc, label='Testing accuracy')
plt.xticks(np.arange(0, num_epochs, step=1), np.arange(1, num_epochs+1, step=1).astype('str').tolist())
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.legend()

plt.savefig('figs/'+'Acc'+figname)
plt.show()