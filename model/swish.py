import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Swish(nn.Module):
    def __init__(self, train_beta=False):
        super(Swish, self).__init__()
        if train_beta:
            self.weight = Parameter(torch.Tensor([1.]))
        else:
            self.weight = 1.0

    def forward(self, input):
        return input * torch.sigmoid(self.weight * input)


def test():
    x = torch.FloatTensor(16, 128, 16, 16)
    swish = Swish(train_beta=True)
    print(swish(x).size())

if __name__ == '__main__':
    test()
