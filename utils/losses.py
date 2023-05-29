import torch.nn as nn


class HogRegressionLoss(nn.Module):
    def __init__(self):
        super(HogRegressionLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, hog_preds, hog_features):
        assert hog_preds.size() == hog_features.size()
        hog_preds = self.log_softmax(hog_preds)
        hog_features = self.softmax(hog_features)

        return self.criterion(hog_preds, hog_features)
