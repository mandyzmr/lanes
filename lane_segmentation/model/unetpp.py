from torch import nn
from torchsummary import summary
from resnet_astrous import *
from config import *


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, n_bridges, up_mode):
        super(UpBlock, self).__init__()
        if up_mode == 'upconv': #转置 (2n-2)/2+1=n
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample': #双线性插入
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_chans, out_chans, kernel_size=1)) #运算后再融合一下features
        
        #用BasicBlock代替ConvBlock，由于有多个bridges通过skip conn加入到Decoder，bridge_chans为多个bridges chans之和
        self.conv_block = BasicBlock(out_chans+(n_bridges*out_chans), out_chans) 
      
    def forward(self, x, bridge_list):
        up = self.up(x)
        # up = transforms.CenterCrop((bridge.size(2),bridge.size(3)))(up) 
        out = torch.cat([up]+bridge_list, dim=1)
        out = self.conv_block(out)
        return out


class ResNet_UNetpp(nn.Module): 
    def __init__(self, backbone='resnet50', up_mode='upconv', l=4, pretrained=False):
        super(ResNet_UNetpp, self).__init__()
        self.l = l #下采样次数
        self.up_mode = 'upconv' #上采样方式
        assert self.up_mode in ('upconv', 'upsample')
        
        # Encoder：没有空洞卷积的16倍下采样ResNet，首层stem不下采样
        self.encoder = get_resnet(backbone, os=16, dilation=[1,1,1], kernel_size=3, stride=1, pretrained=pretrained)
        fea_chans = [64] + [i*self.encoder.block.expansion for i in [64, 128, 256, 512]] #ResNet每层的输出channel
        
        # Decoder：标注x01代表第0层第1个upsample结果，依次根据层深度增加相关结果
        if self.l>=1:
            self.x01 = UpBlock(fea_chans[1], fea_chans[0], 1, self.up_mode)
            self.cls_x01 = nn.Conv2d(fea_chans[0], NUM_CLASSES, kernel_size=1)
        if self.l>=2:
            self.x11 = UpBlock(fea_chans[2], fea_chans[1], 1, self.up_mode)
            self.x02 = UpBlock(fea_chans[1], fea_chans[0], 2, self.up_mode)
            self.cls_x02 = nn.Conv2d(fea_chans[0], NUM_CLASSES, kernel_size=1)
        if self.l>=3:
            self.x21 = UpBlock(fea_chans[3], fea_chans[2], 1, self.up_mode)
            self.x12 = UpBlock(fea_chans[2], fea_chans[1], 2, self.up_mode)
            self.x03 = UpBlock(fea_chans[1], fea_chans[0], 3, self.up_mode)
            self.cls_x03 = nn.Conv2d(fea_chans[0], NUM_CLASSES, kernel_size=1)
        if self.l>=4:
            self.x31 = UpBlock(fea_chans[4], fea_chans[3], 1, self.up_mode)
            self.x22 = UpBlock(fea_chans[3], fea_chans[2], 2, self.up_mode)
            self.x13 = UpBlock(fea_chans[2], fea_chans[1], 3, self.up_mode)
            self.x04 = UpBlock(fea_chans[1], fea_chans[0], 4, self.up_mode)
            self.cls_x04 = nn.Conv2d(fea_chans[0], NUM_CLASSES, kernel_size=1)
           
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #针对Conv relu
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): #针对BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder：首个结果
        x00, x10, x20, x30, x40 = self.encoder(x)
        
        # Decoder
        scores = []
        if self.l>=1:
            x01 = self.x01(x10,[x00]) #从x10上采样后，和bridge list拼接
            out_x01 = self.cls_x01(x01) 
            scores.append(out_x01) #增加每一层的score，后续可以根据所需预测精度对模型适当剪裁
        if self.l>=2:
            x11 = self.x11(x20,[x10]) 
            x02 = self.x02(x11,[x00, x01]) 
            out_x02 = self.cls_x02(x02)
            scores.append(out_x02)
        if self.l>=3:
            x21 = self.x21(x30,[x20]) 
            x12 = self.x12(x21,[x10, x11]) 
            x03 = self.x03(x12,[x00, x01, x02]) 
            out_x03 = self.cls_x03(x03)
            scores.append(out_x03)
        if self.l>=4:
            x31 = self.x31(x40,[x30]) 
            x22 = self.x22(x31,[x20, x21]) 
            x13 = self.x13(x22,[x10, x11, x12]) 
            x04 = self.x04(x13,[x00, x01, x02, x03]) 
            out_x04 = self.cls_x04(x04)
            scores.append(out_x04)
        return scores[-1] #训练时只针对最后一个预测进行优化
   
        # return scores #后期预测时可以修改返回值，根据每层预测的情况，选择性删减模型

if __name__=='__main__':
    model = ResNet_UNetpp('resnet18', up_mode='upconv', l=4)
    summary(model, (3,384,1024))
    