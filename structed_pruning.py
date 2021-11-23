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

def structed_pruning(model, amount, ln_norm, dim):
    for name, module in model.named_modules():
        # prune 20% of connections in all 2D-conv layers
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', n = ln_norm, amount = amount, dim = dim)

def caculate_model_size(model, q_byte = 4):
    # Calculate the pruned weight size:
    total_weight_paras = 0
    linear1_in_channel_reduce = 0
    for name, module in model.named_modules():
        if name == "conv1":
            weight = module.weight
            shape = module.weight.shape
            if dim == 1:
                # channel-wise
                for sub_channel in range(shape[1]):
                    if torch.sum(torch.abs(weight[:,sub_channel,:,:])) != 0:
                        total_weight_paras += shape[0] * shape[2] * shape[3]
                total_weight_paras += shape[1] # Bias
                total_weight_paras += 2 * shape[1] # BatchNorm2d
            else:
                # filter-wise
                for sub_filter in range(shape[0]):
                    if torch.sum(torch.abs(weight[sub_filter,:,:,:])) != 0:
                        total_weight_paras += shape[1] * shape[2] * shape[3]
                        total_weight_paras += 1 # Bias
                        total_weight_paras += 2 # BatchNorm2d
                    else:
                        total_weight_paras -= model.conv2.weight.shape[0] * model.conv2.weight.shape[2] * model.conv2.weight.shape[3] # Remove in-channel for conv2
        if name == "conv2":
            weight = module.weight
            shape = module.weight.shape
            if dim == 1:
                # channel-wise
                for sub_channel in range(shape[1]):
                    if torch.sum(torch.abs(weight[:,sub_channel,:,:])) != 0:
                        total_weight_paras += shape[0] * shape[2] * shape[3]
                else:
                    total_weight_paras -= model.conv1.weight.shape[1] * model.conv1.weight.shape[2] * model.conv2.weight.shape[3] # Remove out-channel for conv1
                    total_weight_paras -= 2 # Remove out-channel for bn1
                total_weight_paras += shape[1] # Bias
                total_weight_paras += 2 * shape[1] # BatchNorm2d
            else:
                # filter-wise
                for sub_filter in range(shape[0]):
                    if torch.sum(torch.abs(weight[sub_filter,:,:,:])) != 0:
                        total_weight_paras += shape[1] * shape[2] * shape[3]
                        total_weight_paras += 1 # Bias
                    else:
                        linear1_in_channel_reduce += 1
                        total_weight_paras += 2 # BatchNorm2d

        if name == "lin1":
            total_weight_paras += 7 * 7 * (64 - linear1_in_channel_reduce) * 7 * 7 * 16
            total_weight_paras += 7 * 7 * 16 # bias
        if name == "lin2":
            total_weight_paras += 7 * 7 * 16 * 10
            total_weight_paras += 10 # bias
    
    return total_weight_paras * q_byte / 1024.0

if __name__ == "__main__":

    # args management
    parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - Task2/3 Quantizations')
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

    MODEL_LOADPATH = "models/model_quantized8.pth"
    # MODEL_UINTX = "models/model_uint"+str(quant_bits)+".pth"
    MODEL_SAVEPATH = "models/model_pruned_" + args.ln_norm + "_" +str(amount)+".pth"

    model = MyConvNet().to(device)
    model.load_state_dict(torch.load(MODEL_LOADPATH))
    
    test_dataset = dsets.MNIST(root ='./data',
                           train = False,
                           transform = transforms.ToTensor(), download=True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)

    structed_pruning(model=model, amount = amount, ln_norm = ln_norm, dim = dim)

    for name, layer in model.named_modules():
        # remove the zeros in layer
        if isinstance(layer, torch.nn.Conv2d):
            prune.remove(layer, name='weight')

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: % f %%' % (100 * correct / total))
    
    print('Model parameter size: %f KB' % (caculate_model_size(model, q_byte=1)))

    # torch.save(model.state_dict(), MODEL_SAVEPATH)
