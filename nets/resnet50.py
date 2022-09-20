from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import bias
from cgitb import reset
from email.mime import base
from email.policy import strict
import math
from numpy import outer

#from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

__all__ = ['ResNet', 'sa_resnet50', 'sa_resnet101', 'sa_resnet152']

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

class NonLocal2d(nn.Module):
    def __init__(self, in_channels, reduction=2, use_scale=True, sub_sample=False, mode='embedded_gaussian'):
        super(NonLocal2d, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode
 
        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')
 
        self.g = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,)
        self.conv_out = nn.Conv2d(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,)
 
        if self.mode != 'gaussian':
            self.theta = nn.Conv2d(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,)
            self.phi = nn.Conv2d(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,)
 
        if self.mode == 'concatenation':
            self.concat_project = nn.Sequential(
                nn.Conv2d(
                    self.inter_channels * 2,
                    1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU())
        self.sub_sample = sub_sample
        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer
 
        self.init_weights()
 
 
 
 
    def init_weights(self, std=0.01, zeros_init=True):
        if self.mode != 'gaussian':
            for m in [self.g, self.theta, self.phi]:
                nn.init.normal_(m.weight.data, std=std)
        else:
            nn.init.normal_(self.g.weight.data, std=std)
            
        if zeros_init:
            nn.init.normal_(self.conv_out.weight.data, 0)
        else:
            nn.init.normal_(self.conv_out.weight.data, std=std)
 
 
 
    def gaussian(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
 
    def embedded_gaussian(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight
 
    def dot_product(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight
 
    def concatenation(self, theta_x, phi_x):
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
 
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]
 
        return pairwise_weight
 
    def forward(self, x):
        # NonLocal2d x: [N, C, H, W]
        n = x.size(0)
        # NonLocal2d g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        if self.mode == 'gaussian':
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == 'concatenation':
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)
 
        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)
 
        # NonLocal2d y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # NonLocal2d y: [N, C, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels,
                                                    *x.size()[2:])
        output = x + self.conv_out(y)
 
        return output
class GC(nn.Module):
    def __init__(self, in_channels, scale = 16):
        super(GC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h*w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        return out
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):#reduction表示缩减率c/r
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class sca_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sca_layer, self).__init__()

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.cweight = Parameter(torch.zeros(1, (channel) // (groups//2), 1, 1))
        self.cbias = Parameter(torch.ones(1, (channel) // (groups//2), 1, 1))
        self.sweight = Parameter(torch.zeros(1, (channel) // (groups//2), 1, 1))
        self.sbias = Parameter(torch.ones(1, (channel) // (groups//2), 1, 1))

        self.fc = nn.Sequential(nn.Conv2d((channel)//(groups//4) ,channel*4 , 1, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d((channel*4), (channel) // (groups//2), 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm((channel) // (groups//2), (channel) // (groups//2))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        
        # channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        xn = self.avg_pool(x_0) + self.max_pool(x_0) + self.max_pool(x_1) + self.avg_pool(x_1)+avg_out+max_out
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1) + self.gn(x_0)#+self.gn(xss)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride=1)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes,stride)
    self.bn2 = nn.BatchNorm2d(planes)
    #self.sa = sa_layer(planes)
    
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    # out = self.ca(out) * out
    #out = self.sa(out)
    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, groups = 1,padding = 1,bias=False,dilation = 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])
        # self.ca1 = ChannelAttention(self.inplanes)
        # self.sa1 = SpatialAttention()
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#.AvgPool2d(7)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.AvgPool2d(2,2),
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.ca1(x) * x
        #x = self.sa1(x) * x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        #
        x = self.fc(x)
        #x = self.dropout(x)
        return x

# def sa_resnet101(pretrained=False):
#     model = _sanet('SANet-101', SABottleneck, [3, 4, 23, 3], pretrained=pretrained)


# def sa_resnet152(pretrained=False):
#     model = _sanet('sSANet-152', SABottleneck, [3, 8, 36, 3], pretrained=pretrained)
#     return model

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4,6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict,strict=False)
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层
    # ----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu,  model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4,  model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier
