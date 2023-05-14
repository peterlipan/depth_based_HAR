import torch
from torch import nn
import torchvision
from ops.basic_ops import AvgConsensus
from torch.nn.init import normal_, constant_


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 backbone='resnet50', frame_per_seg=None,
                 dropout=0.8, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout
        self.num_class = num_class
        self.frame_per_seg = frame_per_seg
        self.partial_bn = partial_bn

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    frame_per_seg:      {}
    dropout_ratio:      {}
        """.format(backbone, self.modality, self.num_segments, self.frame_per_seg, self.dropout)))

        self._prepare_backbone(backbone, num_class)

        self.consensus = AvgConsensus()

    def _prepare_backbone(self, backbone, num_class):

        models = ['resnet50', 'mobilenet_v2', 'mobilenet_v3_small',
                  'shufflenet_v2_x1_5', 'mobilenet_v3_large']
        assert backbone in models
        model = getattr(torchvision.models, backbone)(pretrained=True)
        std = 0.001

        # Reset the last linear layer and dropout
        if 'shufflenet' in backbone or 'resnet' in backbone:
            feature_dim = model.fc.in_features
            if self.dropout == 0:
                model.fc = nn.Linear(in_features=feature_dim, out_features=num_class, bias=True)
                normal_(model.fc.weight, 0, std)
                constant_(model.fc.bias, 0)
            else:
                model.fc = nn.Sequential(nn.Dropout(p=self.dropout, inplace=False),
                                         nn.Linear(in_features=feature_dim, out_features=num_class, bias=True))
                normal_(model.fc[1].weight, 0, std)
                constant_(model.fc[1].bias, 0)

        elif 'mobilenet_v2' in backbone:
            feature_dim = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(p=self.dropout, inplace=False),
                                             nn.Linear(in_features=feature_dim, out_features=num_class, bias=True))
            normal_(model.classifier[1].weight, 0, std)
            constant_(model.classifier[1].bias, 0)

        elif 'mobilenet_v3' in backbone:
            feature_dim = model.classifier[3].in_features
            model.classifier[2] = nn.Dropout(p=self.dropout, inplace=False)
            model.classifier[3] = nn.Linear(in_features=feature_dim, out_features=num_class, bias=True)
            normal_(model.classifier[3].weight, 0, std)
            constant_(model.classifier[3].bias, 0)

        self.backbone = model

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self.partial_bn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])

            elif isinstance(m, nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self.partial_bn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, image):
        # input: Tensor [N, TxCxL, H, W] (L=1)
        num_channels = 3

        # input.view(...) [NxT, C, H, W]
        base_out = self.backbone(image.view((-1, num_channels) + image.size()[-2:]))
        # base_out [NxT, K (num_class)]

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        # base_out after reshape [N, T, K (num_class)]

        output = self.consensus(base_out, dim=1)
        # output [N, K (num_class)]
        return output
