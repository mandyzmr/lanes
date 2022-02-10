from torch import nn
from torchsummary import summary
import torch.utils.model_zoo as model_zoo


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        #增加dilation_rate做空洞卷积，当kernel_size=3, s=1时，若padding=dilation相当于same conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #增加dilation_rate做空洞卷积，当kernel_size=3, s=1时，若padding=dilation相当于same conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet_Astrous(nn.Module):
    def __init__(self, block, layers, os=16, dilation=[1,1,1], kernel_size=7, stride=2):
        super(ResNet_Astrous, self).__init__()
        strides = self.get_strides(os, dilation)
        self.block = block #用于后续获取expansion
        
        # Stem
        self.inplanes = 64 #stem后的channel
        #根据downsample需要，可以用7x7搭配stride=2或者3x3搭配stride=1，做same conv
        #由于可能会生成不同结构，不保留原层名
        self.conv = nn.Conv2d(3, 64, kernel_size, stride, padding=kernel_size//2, bias=False) #2倍downsample
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #4倍downsample
        
        #除了layer1因为有事先pooling降维之外，其余layer都是第一个block的stride=2降维，其余blocks的stride=1
        #保留所有层名，方便后续加载预训练参数
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1]) #8倍downsample
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], 
                                       dilation=16//os) #若8倍下采样，由于不同ResNet结构layer 3的循环次数不同，统一用d=2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3],
                                       #由于不同ResNet结构layer4都是3次循环，可以自定义每次循环的空洞d，如[1,2,1]
                                       dilation=[i*16//os for i in dilation]) #若8倍下采样，那么这层的dilation要乘以2
        
        # 额外重复2次layer 4结构
        self.layer5 = self._make_layer(block, 512, layers[3], stride=1, dilation=[i*16//os for i in dilation])
        self.layer6 = self._make_layer(block, 512, layers[3], stride=1, dilation=[i*16//os for i in dilation])
   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def get_strides(self, os, dilation):
        #output_stride 可选下采样倍数os=8/16，传统resnet都是2**5=32倍downsampling
        #由于至少8倍下采样，即对应layer1-2，所以可以用列表指定剩下3个block的dilation
        if os==16 and dilation==[1,1,1]: #代表用传统的32倍downsample，不用空洞卷积
            strides=[1,2,2,2] #layer1-4的stride
        elif os==16: #若16倍downsample，仅最后layer 4 strides=1，可以用空洞卷积
            strides=[1,2,2,1] 
        elif os==8: #若8倍downsample，最后layer 3-4 strides=1，可以用空洞卷积
            strides=[1,2,1,1] 
        return strides
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        if isinstance(dilation, int): #根据具体循环次数构造
            dilation=[dilation]*blocks  
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): #跳过第一个block
            layers.append(block(self.inplanes, planes, dilation=dilation[i]))
        return nn.Sequential(*layers)    
    
    def forward(self, x):
        stem = self.relu(self.bn1(self.conv(x))) #1/2
        l1 = self.layer1(self.maxpool(stem)) #2/4
        l2 = self.layer2(l1) #4/8
        l3 = self.layer3(l2) #8/16
        x = self.layer4(l3) #8/16/32
        x = self.layer5(x)
        x = self.layer6(x)
        return [stem, l1, l2, l3, x]


def get_resnet(name='resnet101', os=8, dilation=[1,1,1], kernel_size=7, stride=2, pretrained=False):
    structure = {
        'resnet18': (BasicBlock,[2,2,2,2], os, dilation, kernel_size, stride),
        'resnet34': (BasicBlock,[3,4,6,3], os, dilation, kernel_size, stride),
        'resnet50': (Bottleneck,[3,4,6,3], os, dilation, kernel_size, stride),
        'resnet101': (Bottleneck,[3,4,23,3], os, dilation, kernel_size, stride),
        'resnet152': (Bottleneck,[3,8,36,3], os, dilation, kernel_size, stride)}
    
    pretrained_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',}
    
    model = ResNet_Astrous(*structure[name])
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(pretrained_urls[name])
        state_dict = model.state_dict()
        for key, value in pretrained_state_dict.items(): 
            if key in state_dict: #由于新模型会删除一些旧结构，不需要加载多余的参数
                state_dict[key]=pretrained_state_dict[key] #用预训练参数覆盖，空洞卷积不会改变参数量
        model.load_state_dict(state_dict)
    return model


if __name__=='__main__':
    model = get_resnet('resnet101', os=8, dilation=[1,2,4])
    summary(model, (3,384,1024))