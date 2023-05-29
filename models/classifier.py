import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class HogRegress(nn.Module):
    """ MLP for Histogram of the Orientated Gradients """
    def __init__(self, in_features, num_orientations):
        super(HogRegress, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=in_features // 2)
        self.linear2 = nn.Linear(in_features=in_features // 2, out_features=num_orientations)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
