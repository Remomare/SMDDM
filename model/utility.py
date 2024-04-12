import torch.nn as nn

def get_activation_function(activation_fn):
    if activation_fn == "relu":
        return nn.ReLU(inplace=True)
    if activation_fn == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    if activation_fn == "elu":
        return nn.ELU(inplace=True)
    if activation_fn == "swish":
        return nn.SiLU(inplace=True)   
