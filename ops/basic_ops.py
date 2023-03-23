import torch
import torch.nn as nn

class AvgConsensus(nn.Module):

    def __init__(self):
        super(AvgConsensus, self).__init__()
        pass

    def forward(self, input, dim=1):
        assert isinstance(input, torch.Tensor)

        output = input.mean(dim=dim, keepdim=False)
        return output