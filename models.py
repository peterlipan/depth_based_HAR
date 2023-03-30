import torch
from torch import nn
import torchvision
from ops.basic_ops import AvgConsensus
from torch.nn.init import normal_, constant_


class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 backbone='resnet50', new_length=None,
                 dropout=0.8, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.dropout = dropout
        self.num_class = num_class
        self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    dropout_ratio:      {}
        """.format(backbone, self.modality, self.num_segments, self.new_length, self.dropout)))

        self._prepare_backbone(backbone, num_class)

        if self.modality == 'depthDiff':
            self.backbone = self._construct_diff_model(self.backbone)

        self.consensus = AvgConsensus()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

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
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

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
                if not self._enable_pbn or bn_cnt == 1:
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

    def forward(self, input):
        # input: Tensor [N, TxC, H, W]
        sample_len = (3 if self.modality == "depth" else 2) * self.new_length

        if self.modality == 'depthDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # input.view(...) [NxT, C, H, W]
        base_out = self.backbone(input.view((-1, sample_len) + input.size()[-2:]))
        # base_out [NxT, K (num_class)]

        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        # base_out after reshape [N, T, K (num_class)]

        output = self.consensus(base_out, dim=1)
        # output [N, K (num_class)]
        return output

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["depth", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, backbone):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return backbone

    def _construct_diff_model(self, backbone, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return backbone
