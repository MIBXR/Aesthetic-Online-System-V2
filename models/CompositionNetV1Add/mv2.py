import os
import math
import torch
import torch.nn as nn
from collections import OrderedDict
# import sys
# sys.path.append("..")
# print(sys.path)

MOBILE_NET_V2_UTR = 'https://s3-us-west-1.amazonaws.com/models-nima/mobilenetv2.pth.tar'

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)


        #avgpool
        self.avgpool = nn.AvgPool2d(input_size // 32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class),
        )

        # 对网络中的Conv2d层、BatchNorm2d层、Linear层进行统一初始化
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobile_net_v2(pretrained=False):
    model = MobileNetV2()
    return model

def SelfAttentionMap(x):
    batch_size, in_channels, h, w = x.size()
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    quary = quary.permute(0, 2, 1)

    sim_map = torch.matmul(quary, key)

    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

def Back_Net(pretrained = False):
    model = mobile_net_v2()
    model = nn.Sequential(*list(model.children())[:-1])

    # 去除avgpool
    #model = nn.Sequential(*list(model.children())[:-2])
    # print(model)
    model_dict = model.state_dict()

    if pretrained:
        print("read Back_Net weights")
        # path_to_model = "./models/pretrain_model/back_model.pth"
        path_to_model = "./noise_back_e_new2021_01_16_13_11_epoch_11_vacc0.8288845401174169_tacc0.8180964308077645.pth"
        # 将模型从GPU->CPU https://www.dazhuanlan.com/2019/12/25/5e0263da3c913/
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)

        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if k[:11] == 'base_model.':
                name = k[11:]
            else:
                name= k
            new_state_dict[name] = v
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict)
    return model

def Fore_Net(pretrained = True):
    model = mobile_net_v2()
    model = nn.Sequential(*list(model.children())[:-2])
    model_dict = model.state_dict()
    if pretrained:
        print("read Fore_Net weights")
        path_to_model = "./models/pretrain_model/fore_model.pth"
        path_to_model = "./noise_fore_u_new2021_01_21_13_12_epoch_35_vacc0.8251272015655577_tacc0.8195835942391985.pth"

        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if k[:11] == 'base_model.':
                name = k[11:]
            else:
                name= k
            new_state_dict[name] = v
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model.load_state_dict(pretrained_dict)

    return model

class CAT(nn.Module):
    def __init__(self):
        super(CAT,self).__init__()
        back_model = Back_Net(pretrained = True)
        fore_model = Fore_Net(pretrained = True)

        # back_model = Back_Net(pretrained = False)
        # fore_model = Fore_Net(pretrained = False)

        self.base_model = back_model
        self.sa_model = fore_model

        ### 需要加的  36 * (H1 * W1 H2 * W2)
        self.adapool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adapool2 = nn.AdaptiveAvgPool2d((49, 49))

    def forward(self, x):
        x_base = self.base_model(x)  # x_shape:torch.Size([32, 3, 224, 224])
        # print("x_shape:{0}".format(x.shape))
        x_sa = self.sa_model(x)  # ([32, 1280, 7, 7])
        # print("x_sa_shape:{0}".format(x_sa.shape))
        x_sa = SelfAttentionMap(x_sa)  # torch.Size([32, 49, 49])
        # print("x_SelfAttentionMap_shape:{0}".format(x_sa.shape))



        #x_sa = x_sa.view(-1, 1280, 49)
        # print("x_sa_shape:{0}".format(x_sa.shape))

        x_base = self.adapool1(x_base)
        x_sa = self.adapool2(x_sa)

        x = x_base.view(x_base.size(0),-1)
        x1 = x_sa.view(x_sa.size(0),-1)

        return x,x1

def cat_net():
    model = CAT()
    return model