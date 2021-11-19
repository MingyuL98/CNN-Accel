import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

from ConvNet import MyConvNet
import argparse

Q_Tensor = namedtuple('Q_Tensor', ['tensor', 'scale', 'zero'])

def quantize_tensor(x, num_bits):
    x_min, x_max = x.min(), x.max()
    q_min = 0.
    q_max = 2.**num_bits - 1

    scale = (x_max - x_min) / (q_max - q_min)
    zero_init = q_min - x_min / scale
    
    q_zero = 0
    if zero_init < q_min:
        q_zero = q_min
    elif zero_init > q_max:
        q_zero = q_max
    else:
        q_zero = zero_init

    q_zero = int(q_zero)
    q_x = q_zero + x / scale
    q_x.clamp_(q_min, q_max)
    # q_x = q_x.to(torch.uint8)
    # q_x = q_x.to(torch.int16)
    q_x = q_x.to(torch.int32)
    return Q_Tensor(tensor=q_x, scale=scale, zero=q_zero)   

def dequantize_tensor(x):
    return x.scale * (x.tensor.float() - x.zero)

def quantize_model(model, num_bits):
    q_tensor_params = {}

    for name, param in model.state_dict().items():
        q_param = quantize_tensor(param, num_bits)
        q_tensor_params[name+'-quant-scale'] = torch.Tensor([q_param.scale])
        # q_tensor_params[name+'-quant-zero'] = torch.ShortTensor([q_param.zero]) 
        # q_tensor_params[name+'-quant-zero'] = torch.ByteTensor([q_param.zero])
        q_tensor_params[name+'-quant-zero'] = torch.Tensor([q_param.zero])

        param.copy_(q_param.tensor)

    model.type('torch.Tensor')
    # model.type('torch.ShortTensor')
    # model.type('torch.ByteTensor')
    for name, param in q_tensor_params.items():
        name = name.translate({46: 45})
        model.register_buffer(name, param)
    model.quant = True
    
def dequantize_model(model, num_bits):
    model.type('torch.Tensor')
    model_params = model.state_dict()

    for name, param in model_params.items():
        if 'quant' not in name:
            name = name.translate({46: 45})
            q_param = Q_Tensor(tensor=param,
                               scale=model_params[name+'-quant-scale'][0],
                               zero=model_params[name+'-quant-zero'][0])
            param.copy_(dequantize_tensor(q_param))

            model.register_buffer(name+'-quant-scale', None)
            model.register_buffer(name+'-quant-zero', None)

    model.quant = None

if __name__ == "__main__":

    # args management
    parser = argparse.ArgumentParser(description='ML_CODESIGN Lab3 - Task2/3 Quantizations')
    parser.add_argument('--num-bits', type=int, default=8, help='Quantization bits')
    parser.add_argument('--print', type=int, default=0, help='Print the first layer')

    args = parser.parse_args()

    # params
    quant_bits = args.num_bits
    do_printing = args.print
    represent_bits = 32

    MODEL_LOADPATH = "models/task1.pth"
    MODEL_UINTX = "models/model_uint"+str(quant_bits)+".pth"
    MODEL_SAVEPATH = "models/model_quantized"+str(quant_bits)+".pth"

    model = MyConvNet()
    model.load_state_dict(torch.load(MODEL_LOADPATH))
    
    if (do_printing):
        for param in model.parameters():
            print(param.data)
            break
    
    quantize_model(model, num_bits=quant_bits)

    if (do_printing):
        for param in model.parameters():
            print(param.data)
            break
    
    torch.save(model.state_dict(), MODEL_UINTX)

    dequantize_model(model, num_bits=represent_bits)

    if (do_printing):
        for param in model.parameters():
            print(param.data)
            break
    
    torch.save(model.state_dict(), MODEL_SAVEPATH)
