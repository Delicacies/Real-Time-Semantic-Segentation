##############################################################
# Created by: YJY
# Date: 2020-03-27
# Copyright (c) 2017
##############################################################

"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn,functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module): #继承父类nn.Module
    def __init__(self, num_classes, aux=False, **kwargs): # **kwargs为未知内容的形参
        super(FastSCNN, self).__init__()# FastSCNN是一个nn.Moudle的一个子类，self是其一个实例，
                                        # super把self转化为父类nn.Moudle的实例，然后使用init()方法
        self.aux = aux #auxiliary：添加辅助
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeartureExtractor(64, [64, 96,128], 128, 6, [3, 3, 3])
        self.feature_fusion = FuatureFusionModule)(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential( # nn.Sequential需逗号隔开
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLu(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self,x):
        size = x.size()[2:] # x.size()=n, c, h, w；[2:]表示切片，从size()的第3个值开始取值
        higher_res_features = self.learning_to_downsample(x) # 学习下采样，浅层特征提取
        x = self.global_feature_extractor(higher_res_features) # 深层全局特征提取
        x = self.feature_fusion(higher_res_features, x) # 浅层特征与深层特征融合
        x = classifier(x) # 像素分类器
        outputs = []
        x = F.interplotate(x, size, mode='bilinear', align_corners=True) # 双线性插入恢复图像尺寸，效果一般但速度快
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interplotate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""
    """ in_channels: 输入通道数（层数）
        out_channels: 输出通道数（层数）
        kernal_size: 卷积核尺寸（3×3 or 3×5）
        stride: cross-correlation的步长，卷积核滑动时每次跨越的像素个数
        padding：四边补0，padding=0表示不需补0
        dilation: 膨胀率，卷积核点间的距离
        groups: 卷积核个数，
        bias: 偏置，每个像素值加上偏置值
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


    
class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Moudle):
    def __init__(self, dw_channels,out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2,  **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride ==1 and in_channels ==out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw - linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size):
        return avgpool(x)

    def upsammple(self, x, size):
        return F.interplotate(x, size, mode='bilinear', align_corner=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsammple(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsammple(self.conv2(self.pool(x, 1)), size)
        feat3 = self.upsammple(self.conv3(self.pool(x, 1)), size)
        feat4 = self.upsammple(self.conv4(self.pool(x, 1)), size)
         x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
         x = self.out(x)
         return x


    class LearningToDownsample(nn.Module):
        """Learning to downsample module"""

        def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
            super(LearningToDownsample, self).__init__()
            self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
            self.dsconv1 = _DSConv(dw_channel1, dw_channels2, 2)
            self.dsconv12 = _DSConv(dw_channels2, out_channels, 2)

        def forward(self, x):
            x = self.conv(x)
            x = self.dsconv1(x)
            x = self.dsconv2(x)
            return x


    class GlobalFeartureExtractor(nn.Moudle):
        """Global feature extractor module"""

        def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, t=6, num_blocks=(3,3,3), **kwargs):
            super(GlobalFeartureExtractor, self).__init__()
            self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
            self.bottleneck2 = self._make_layer(LinearBottleneck, in_channels, block_channels[1], num_blocks[1], t, 2)
            self.bottleneck3 = self._make_layer(LinearBottleneck, in_channels, block_channels[2], num_blocks[2], t, 2)
            self.ppm = PyramidPooling(block_channels[2], out_channels)

        def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
            layers = []
            layers.append(block(inplanes, planes, t, stride))
            for i in range(1, blocks):
                layers.append(block(planes, planes, t, 1))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.bottleneck1(x)
            x = self.bottleneck2(x)
            x = self.bottleneck3(x)
            return x


    class FeatureFusionModule(nn.Module):
        """Feature fusion module"""

        def __init__(self, higher_in_channels, lower_in_channels, out_channel, sacle_factor=4, **kwargs):
            super(FeatureFusionModule, self).__init__()
            self.sacle_factor = sacle_factor
            self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
            self.conv_lower_res = nn.Sequential(
                nn.Conv2d(higher_in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.conv_higher_res = nn.Sequential(
                nn.Conv2d(higher_in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.relu = nn.ReLU(True)

        def forward(self, higher_res_feature, lower_res_feature):
            lower_res_feature = F.interplotate(lower_res_feature, sacle_factor=4, mode='bilinear', align_corner=True)
            lower_res_feature = self.dwconv(lower_res_feature)
            lower_res_feature = self.conv_lower_res(lower_res_feature)

            higher_res_feature = self.conv_higher_res(higher_res_feature)
            out = higher_res_feature + lower_res_feature
            return self.relu(out)


    class Classifer(nn.Module):
        """Classifer"""

        def __init__(self, dw_channels, num_classes, stride=1, **kwarga):
            super(Classifer, self).__init__()
            self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
            self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
            self.conv = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(dw_channels, num_classes, 1)
            )

        def forward(self, x):
            x = self.dsconv1(x)
            x = self.dsconv2(x)
            x = self.conv(x)
            return x


    def get_fast_scnn(dataset='citys', pretraied=False, root='./weights', map_cpu=False, **kwargs):
        acronyms = {
            'pascal_voc': 'voc',
            'pascal_aug': 'voc',
            'ade20k': 'ade',
            'coco': 'coco',
            'citys': 'citys',
        }
        from dataloader import datasets
        model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
        if pretraied:
            if(map_cpu):
                model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
            else:
                model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]))) #'fast_scnn_citys (tjark_600_epoch).pth'
        return module



    if __name__ == '__main__':
        img = torch.randn(2. 3, 256, 512)
        model = get_fast_scnn('citys')
        outputs = model(img)

