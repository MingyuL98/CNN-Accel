import argparse
import numpy as np
import torch
import torch.nn as nn
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

if __name__ == "__main__":

    # args management
    parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - Task2/3 Quantizations')
    parser.add_argument('--pruning_type', type=str, choices=["channel", "filter"], default="channel", help='Pruning Type')
    parser.add_argument('--amount', type=float, default=0.2, help='Pruning amount')
    parser.add_argument('--ln_norm', type=str, choices=["l1", "l2"], default="l2", help='Pruning Norm')
    args = parser.parse_args()

    # params
    batch_size = 64
    ln_norm = 1 if args.ln_norm == "l1" else 2
    dim = 0 if args.pruning_type == "filter" else 1
    amount = args.amount

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_LOADPATH = "models/task1.pth"
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
    summary(model, input_size=(1, 1, 28, 28))
    # torch.save(model.state_dict(), MODEL_SAVEPATH)
    