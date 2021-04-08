import torch.nn as nn
import torch


class BasicBlock(nn.Module):  # 定义18层、34层残差网络
    expansion = 1         # 定义卷积层是否改变，1表示不变
    #定义初始函数，定义残差网络所需要使用的一系列层结构

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #两个3x3卷积层，注意在第一个中由于实线虚线不同的区别
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        #定义前向传播
    def forward(self, x):
        identity = x
        # 如果是虚线需要进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):  # 定义50层、101层、152层残差网络
        expansion = 4         # 定义卷积层是否改变，4表示conv3是conv2的4倍
        # 定义初始函数，定义残差网络所需要使用的一系列层结构

        def __init__(self, in_channel, out_channel, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            # 3个卷积层，注意在第一个中由于实线虚线不同的区别
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)

            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                   kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                                   kernel_size=1, stride=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)  #？？？需要乘以4吗
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
        #定义前向传播

        def forward(self, x):
            identity = x
            #print(x.shape)
            # 如果是虚线需要进行下采样
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            #print(out.shape)
            out += identity
            out = self.relu(out)
            return out

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=10, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            #自适应平均池化
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        #剩下的全是实线
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)
        #将list列表转换为一个非关键字参数

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            #print(x.shape)
        return x

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

