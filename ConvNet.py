import torch
import torch.nn as nn
from torchsummary import summary    # for model summary
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

import argparse
import matplotlib.pyplot as plt
import numpy as np

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin1 = nn.Linear(7*7*64, 7*7*16)
        self.lin2 = nn.Linear(7*7*16, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        a1 = self.act1(b1)
        p1 = self.pool1(a1)
        c2 = self.conv2(p1)
        b2 = self.bn2(c2)
        a2 = self.act2(b2)
        p2 = self.pool2(a2)
        flt = p2.view(p2.size(0), -1)
        l1 = self.lin1(flt)
        out = self.lin2(l1)
        return out


if __name__ == "__main__":

    # args management
    parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - Task2 Quantization Effects')
    parser.add_argument('--test', type=int, default=0, help='Test the model')
    parser.add_argument('--draw-hist', type=int, default=0, help='Histograms of the weights from critical layers')
    parser.add_argument('--model-sel', type=int, default=0, help='0: initial model, 1: quantized model, 2: reserved etc')   
    parser.add_argument('--num-bits', type=int, default=8, help='Number of quantization bits')   

    args = parser.parse_args()

    # parsing
    do_testing = args.test
    draw_hist = args.draw_hist
    model_sel = args.model_sel

    # params
    batch_size = 64
    MODEL_PATH_F32 = "models/task1.pth"
    MODEL_PATH_UINTX = "models/model_quantized"+str(args.num_bits)+".pth"

    # init model
    model = MyConvNet()
    # summary(model, (1, 28, 28))

    # Quantization effects
    if(do_testing):

        # before quantization
        if (model_sel == 0): 
            model.load_state_dict(torch.load(MODEL_PATH_F32))
        elif (model_sel == 1):
            model.load_state_dict(torch.load(MODEL_PATH_UINTX))

        if (draw_hist):
            # plot weight histogram (conv & linear layers)
            fig1, axs1 = plt.subplots(2, 2, tight_layout=True)
            x_num = 0
            y_num = 0

            for layer in model.modules():
                if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                    print(layer)
                    x = np.array(layer.weight.reshape(1,-1).cpu().detach())
                    counts, bins = np.histogram(x)
                    axs1[x_num][y_num].hist(bins[:-1], bins, weights=counts)

                    y_num += 1
                    if(y_num > 1):
                        y_num = 0
                        x_num += 1       

            axs1[0][0].set_title('conv1')
            axs1[0][1].set_title('conv2')
            axs1[1][0].set_title('lin1')
            axs1[1][1].set_title('lin2')
            plt.show()
               

        # load test dataset
        test_dataset = dsets.MNIST(root ='./data',
                                    train = False,
                                    transform = transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False)

        # test the accuracy
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            correct = 0.0
            total = 0.0

            for images, labels in test_loader:
                images = Variable(images)
                labels = labels
                outputs = model(images)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)

                correct += (predicted == labels).sum()

            acc = 100 * float(correct)/total

            print('Accuracy of the model on the 10000 test images: % .4f %%' % acc)
            print('Loss of the model on the 10000 test images: %.4f'  % (loss))
