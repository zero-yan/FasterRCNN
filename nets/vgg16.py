import torch
import torch.nn as nn
#from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

from torch.nn.parameter import Parameter
class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
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
#--------------------------------------#
#   VGG16的结构
#--------------------------------------#
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.inplanes = 64
        self.features = features
        #--------------------------------------#
        #   平均池化到7x7大小
        #--------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.dropout = nn.Dropout(0.5)
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        #--------------------------------------#
        #   特征提取
        #--------------------------------------#
        x = self.features(x)
        #--------------------------------------#
        #   平均池化
        #--------------------------------------#
        x = self.avgpool(x)
        #--------------------------------------#
        #   平铺后
        #--------------------------------------#
        x = torch.flatten(x, 1)
        #x = self.dropout(x)
        #--------------------------------------#
        #   分类部分
        #--------------------------------------#
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

'''
假设输入图像为(600, 600, 3)，随着cfg的循环，特征层变化如下：
600,600,3 -> 600,600,64 -> 600,600,64 -> 300,300,64 -> 300,300,128 -> 300,300,128 -> 150,150,128 -> 150,150,256 -> 150,150,256 -> 150,150,256 
-> 75,75,256 -> 75,75,512 -> 75,75,512 -> 75,75,512 -> 37,37,512 ->  37,37,512 -> 37,37,512 -> 37,37,512
到cfg结束，我们获得了一个37,37,512的特征层
'''

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

#--------------------------------------#
#   特征提取部分
#--------------------------------------#
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            #SElayer(64)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            s = v * 4
            if batch_norm:
                
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True),SElayer(s)]
            else:
                layers += [conv2d,SElayer(v),nn.ReLU(inplace=True)]#,ChannelAttention(v),SpatialAttention(v)]
            in_channels = v
            
    return nn.Sequential(*layers)

def decom_vgg16(pretrained = False):
    model = VGG(make_layers(cfg))
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict,strict=False)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，最终获得一个37,37,1024的特征层
    #----------------------------------------------------------------------------#
    features    = list(model.features)[:30]
    #----------------------------------------------------------------------------#
    #   获取分类部分，需要除去Dropout部分
    #----------------------------------------------------------------------------#
    classifier  = list(model.classifier)
    del classifier[6]
    del classifier[5]
    del classifier[2]

    features    = nn.Sequential(*features)
    classifier  = nn.Sequential(*classifier)
    return features, classifier
