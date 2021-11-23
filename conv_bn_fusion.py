import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import parameter
import torch.nn.utils.prune as prune
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchinfo import summary
from ConvNet import MyConvNet

class FusedConvNet(nn.Module):
    def __init__(self, conv1_out_channel, conv2_out_channel):
        super(FusedConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_channel, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(conv1_out_channel, conv2_out_channel, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.lin1 = nn.Linear(7*7*conv2_out_channel, 7*7*16)
        self.lin2 = nn.Linear(7*7*16, 10)

    def forward(self, x):
        c1 = self.conv1(x)
        a1 = self.act1(c1)
        p1 = self.pool1(a1)
        c2 = self.conv2(p1)
        a2 = self.act2(c2)
        p2 = self.pool2(a2)
        flt = p2.view(p2.size(0), -1)
        l1 = self.lin1(flt)
        out = self.lin2(l1)
        return out

def fuse_conv_bn1(conv_layer, bn_layer, device):
    conv_weight = []
    conv_bias = []
    bn_mean = []
    bn_running_var = []
    bn_beta = []
    bn_gamma = []
    conv1_out_channel_enable = []

    shape = conv_layer.weight.shape
    bn_eps = bn_layer.eps
    
    for sub_filter in range(shape[0]):
        if torch.sum(torch.abs(conv_layer.weight[sub_filter,:,:,:])) != 0:
            conv_weight.append(conv_layer.weight[sub_filter, :, :, :])
            conv_bias.append(conv_layer.bias[sub_filter])
            bn_mean.append(bn_layer.running_mean[sub_filter])
            bn_running_var.append(bn_layer.running_var[sub_filter])
            bn_beta.append(bn_layer.weight[sub_filter])
            bn_gamma.append(bn_layer.bias[sub_filter])
            conv1_out_channel_enable.append(True)
        else:
            conv1_out_channel_enable.append(False)
    
    conv_weight = torch.stack(conv_weight, 0).to(device)
    conv_bias = torch.as_tensor(conv_bias).to(device)
    bn_mean = torch.as_tensor(bn_mean).to(device)
    bn_running_var = torch.as_tensor(bn_running_var).to(device)
    bn_beta = torch.as_tensor(bn_beta).to(device)
    bn_gamma = torch.as_tensor(bn_gamma).to(device)

    sorted_var = torch.sqrt(bn_running_var + bn_eps)
    conv_weight = conv_weight * ((bn_beta / sorted_var).reshape([conv_weight.shape[0], 1, 1, 1]))
    conv_bias = (conv_bias - bn_mean) / sorted_var * bn_beta + bn_gamma
    
    fused_conv_weight = parameter.Parameter(conv_weight)
    fused_conv_bias = parameter.Parameter(conv_bias)

    return fused_conv_weight, fused_conv_bias, conv1_out_channel_enable

def fuse_conv_bn2(conv_layer, bn_layer, conv1_out_channel_enable, device):
    conv_weight = []
    conv_bias = []
    bn_mean = []
    bn_running_var = []
    bn_beta = []
    bn_gamma = []

    shape = conv_layer.weight.shape
    bn_eps = bn_layer.eps
    
    for sub_filter in range(shape[0]):
        if torch.sum(torch.abs(conv_layer.weight[sub_filter,:,:,:])) != 0:
            conv_sub_channel = []
            for sub_channel in range(shape[1]):
                if conv1_out_channel_enable[sub_channel]:
                    conv_sub_channel.append(conv_layer.weight[sub_filter, sub_channel, :, :])
            conv_sub_channel = torch.stack(conv_sub_channel, 0)
            conv_weight.append(conv_sub_channel)
            conv_bias.append(conv_layer.bias[sub_filter])
            bn_mean.append(bn_layer.running_mean[sub_filter])
            bn_running_var.append(bn_layer.running_var[sub_filter])
            bn_beta.append(bn_layer.weight[sub_filter])
            bn_gamma.append(bn_layer.bias[sub_filter])
    
    conv_weight = torch.stack(conv_weight, 0).to(device)
    conv_bias = torch.as_tensor(conv_bias).to(device)
    bn_mean = torch.as_tensor(bn_mean).to(device)
    bn_running_var = torch.as_tensor(bn_running_var).to(device)
    bn_beta = torch.as_tensor(bn_beta).to(device)
    bn_gamma = torch.as_tensor(bn_gamma).to(device)

    sorted_var = torch.sqrt(bn_running_var + bn_eps)
    conv_weight = conv_weight * ((bn_beta / sorted_var).reshape([conv_weight.shape[0], 1, 1, 1]))
    conv_bias = (conv_bias - bn_mean) / sorted_var * bn_beta + bn_gamma
    
    fused_conv_weight = parameter.Parameter(conv_weight)
    fused_conv_bias = parameter.Parameter(conv_bias)

    return fused_conv_weight, fused_conv_bias

def cal_fc1_paras(conv2, lin1, device):
    
    fc_weight = []

    shape = conv2.weight.shape

    for sub_filter in range(shape[0]):
        if torch.sum(torch.abs(conv2.weight[sub_filter,:,:,:])) != 0:
            base = sub_filter * 7 * 7
            for index in range(7 * 7):
                fc_weight.append(lin1.weight[:,base + index])
    
    fc_weight = torch.stack(fc_weight, 1).to(device)

    fc_weight = parameter.Parameter(fc_weight)

    return fc_weight, lin1.bias

def fuse_model(model, device):
    conv1_out_channel = 0
    conv2_out_channel = 0

    for sub_filter in range(model.conv1.weight.shape[0]):
        if torch.sum(torch.abs(model.conv1.weight[sub_filter,:,:,:])) != 0:
            conv1_out_channel += 1

    for sub_filter in range(model.conv2.weight.shape[0]):
        if torch.sum(torch.abs(model.conv2.weight[sub_filter,:,:,:])) != 0:
            conv2_out_channel += 1

    new_model = FusedConvNet(conv1_out_channel, conv2_out_channel).to(device)
    new_model.conv1.weight, new_model.conv1.bias, conv1_out_channel_enable = fuse_conv_bn1(model.conv1, model.bn1, device)
    new_model.conv2.weight, new_model.conv2.bias = fuse_conv_bn2(model.conv2, model.bn2, conv1_out_channel_enable, device)
    new_model.lin1.weight, new_model.lin1.bias = cal_fc1_paras(model.conv2, model.lin1, device)
    new_model.lin2.weight = model.lin2.weight
    new_model.lin2.bias = model.lin2.bias
    return new_model

if __name__ == "__main__":

    # args management
    parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - Task4 Layer Fusion')
    parser.add_argument('--pruning_type', type=str, choices=["channel", "filter"], default="filter", help='Pruning Type')
    parser.add_argument('--amount', type=float, default=0.5, help='Pruning amount')
    parser.add_argument('--ln_norm', type=str, choices=["l1", "l2"], default="l2", help='Pruning Norm')
    args = parser.parse_args()

    # params
    batch_size = 64
    ln_norm = 1 if args.ln_norm == "l1" else 2
    dim = 0 if args.pruning_type == "filter" else 1
    amount = args.amount

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_LOADPATH = "models/model_pruned_l2_0.59.pth"
    # MODEL_UINTX = "models/model_uint"+str(quant_bits)+".pth"
    # MODEL_SAVEPATH = "models/model__" + args.ln_norm + "_" +str(amount)+".pth"

    model = MyConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_LOADPATH))

    fused_model = fuse_model(model, device)
    
    test_dataset = dsets.MNIST(root ='./data',
                           train = False,
                           transform = transforms.ToTensor(), download=True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).to(device)
        labels = labels.to(device)
        outputs = fused_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: % f %%' % (100 * correct / total))
    
