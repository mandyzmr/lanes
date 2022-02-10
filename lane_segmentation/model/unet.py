from torch import nn
from torchsummary import summary
from resnet_astrous import *
from config import *


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, bridge_chans, up_mode):
        super(UpBlock, self).__init__()
        if up_mode == 'upconv': #转置
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample': #双线性插入
            self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(in_chans, out_chans, kernel_size=1)) #运算后再融合一下features
        
        #用BasicBlock代替之前的ConvBlock，不需要定义padding，out_chans和Brige_chans不一定相等，需要分开指定
        self.conv_block = BasicBlock(out_chans+bridge_chans, out_chans) 
      
    def forward(self, x, bridge):
        up = self.up(x)
        crop = transforms.CenterCrop((up.size(2),up.size(3)))(bridge) 
        out = torch.cat([crop, up], dim=1)
        out = self.conv_block(out)
        return out


class ResNet_UNet(nn.Module):
    def __init__(self, backbone='resnet50', up_mode='upconv', pretrained=False):
        super(ResNet_UNet, self).__init__()
        self.up_mode = 'upconv' #上采样方式
        assert self.up_mode in ('upconv', 'upsample')

        # Encoder：没有空洞卷积的16倍下采样ResNet，首层stem不下采样
        self.encoder = get_resnet(backbone, os=16, dilation=[1,1,1], kernel_size=3, stride=1, pretrained=pretrained)
        in_chans = 512 * self.encoder.block.expansion #ResNet最后一层的输出channel
        
        # Decoder 
        self.decoder = nn.ModuleList()
        for i in range(3): #16倍上采样
            self.decoder.append(UpBlock(in_chans, in_chans//2, in_chans//2, self.up_mode))
            in_chans //= 2 #(256, 128, 64)*4
        #最后一步in&out都是64，不是in_chans//2，所以UpBlock里拼接之后的in_chans和原来in_chans不一定相等
        self.decoder.append(UpBlock(in_chans, 64, 64, self.up_mode)) 
        self.cls_conv = nn.Conv2d(64, NUM_CLASSES, kernel_size=1)
        
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #针对Conv relu
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): #针对BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        f1, f2, f3, f4, f5 = self.encoder(x)
        bridges = [f1, f2, f3, f4]
        x = f5
        
        # Decoder
        for i, decode_layer in enumerate(self.decoder):
            x = decode_layer(x, bridges[-i-1]) #逆序取出
        
        x = self.cls_conv(x)
        score = nn.Softmax(dim=1)(x)
        return score


if __name__=='__main__':
    model = ResNet_UNet('resnet18', up_mode='upconv')
    summary(model, (3,384,1024))
    