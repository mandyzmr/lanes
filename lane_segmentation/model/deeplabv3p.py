from torch import nn
from torchsummary import summary
from resnet_astrous import *
from config import *


class ASPP(nn.Module):
    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        kernel_size=[1,3,3,3]
        dilation = [rate*d for d in [1,6,12,18]]
        
        # 并联结构：对ResNet结果用不同d分别做空洞卷积，从而以多个比例感受野捕捉图像的上下文 (14,14,2048)->(14,14,256)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size[0], padding=0, dilation=dilation[0], bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size[1], padding=dilation[1], dilation=dilation[1], bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size[2], padding=dilation[2], dilation=dilation[2], bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size[3], padding=dilation[3], dilation=dilation[3], bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))
        
        # 图像层级特征 (1,1,2048)->(1,1,256)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))
        
        # 合并4个空洞卷积结果和1个gpa结果后，再做一次1x1卷积
        self.branch6 = nn.Sequential(
            nn.Conv2d(out_chans*5, out_chans, 1, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.shape
        c1 = self.branch1(x) #(14,14,256)
        c2 = self.branch2(x)
        c3 = self.branch3(x)
        c4 = self.branch4(x)
        gap = self.branch5(self.avgpool(x))
        gap = nn.Upsample(size=(h,w), mode='bilinear', align_corners=False)(gap) # 上采样到原始inputs大小
        x = torch.cat([c1,c2,c3,c4,gap], dim=1) #(14,14,256*5)
        x = self.branch6(x) #(14,14,256)
        return x


class DeeplabV3p(nn.Module):
    def __init__(self, backbone='resnet101', os=16, dilation=[1,2,4], pretrained=False):
        super(DeeplabV3p, self).__init__()
        
        #L4 深层feature 从原图downsample 8/16倍 -> upsample 2/4倍
        self.backbone = get_resnet(backbone, os, dilation, pretrained=pretrained) #(224,224,3)->(14,14,2048)
        self.aspp = ASPP(512*self.backbone.block.expansion, ASPP_DIM, rate=16//os) #(14,14,2048)->(14,14,256)
        self.dropout = nn.Dropout(0.5) 
        self.upsample1 = nn.Upsample(scale_factor=os//4, mode='bilinear', align_corners=False)  #(56,56,256)
        
        #L1 浅层feature 从原图downsample 4倍 (56,56,64)->(56,56,48)
        self.shortcut = nn.Sequential( #1x1的same conv
                nn.Conv2d(64*self.backbone.block.expansion, SHORTCUT_DIM, SHORTCUT_KERNEL, padding=SHORTCUT_KERNEL//2, bias=False),
                nn.BatchNorm2d(SHORTCUT_DIM),
                nn.ReLU(inplace=True))
        
        # 合并feature (56,56,256+48)->(56,56,256) 
        self.concat = nn.Sequential(
                nn.Conv2d(ASPP_DIM+SHORTCUT_DIM, ASPP_DIM, 3, padding=1, bias=False),
                nn.BatchNorm2d(ASPP_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(ASPP_DIM, ASPP_DIM, 3, padding=1, bias=False),
                nn.BatchNorm2d(ASPP_DIM),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1))
        
        # 根据融合feature，分类并upsample 4倍 (56,56,256)->(224,224,8)
        self.classifier = nn.Conv2d(ASPP_DIM, NUM_CLASSES, 1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 深层features
        stem,l1,l2,l3,l4 = self.backbone(x) 
        feature_aspp = self.aspp(l4)
        feature_aspp = self.dropout(feature_aspp)
        feature_aspp = self.upsample1(feature_aspp)

        # 浅层features
        feature_shallow = self.shortcut(l1)
        
        # 合并features
        feature = torch.cat([feature_aspp, feature_shallow], dim=1)
        x = self.concat(feature)
        x = self.classifier(x)
        x = self.upsample2(x)
        return x

if __name__=='__main__':
    model = DeeplabV3('resnet50', os=16, dilation=[1,2,4])
    summary(model, (3,384,1024))
