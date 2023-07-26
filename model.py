import torch
import torch.nn as nn
import math

class Perceptron(nn.Module):
    def __init__(self, args):
        super(Perceptron, self).__init__()
        self.N = args.N
        self.register_buffer("weight", torch.empty(size=(self.N, )))
        self.resetParameters()

    def resetParameters(self):
        nn.init.normal_(self.weight, 0, 1)

    def forward(self, inputs):
        return torch.sign(torch.mv(inputs, self.weight))
    