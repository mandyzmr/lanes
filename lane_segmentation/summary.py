from model.deeplabv3p import *
from model.unet import *
from model.unetpp import *
from config import *


def get_model(name='deeplabv3p', backbone='resnet101', os=16, dilation=[1,2,4], pretrained=False, unet_layer=4):
    if name=='deeplabv3p':
        model = DeeplabV3p(backbone, os, dilation, pretrained)
    elif name=='unet':
        model = ResNet_UNet(backbone, up_mode='upconv', pretrained=pretrained)
    elif name=='unetpp':
        model = ResNet_UNetpp(backbone, up_mode='upconv', l=unet_layer, pretrained=pretrained)
    return model


def get_pretrained_model(name='deeplabv3p', backbone='resnet101', pretrained_path=None):
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{name}_{backbone}"
    checkpoint_dir = os.path.join(out_dir, model_name, 'checkpoint')
         
    model = get_model(name, backbone, OUTPUT_STRIDE, DILATION, PRETRAINED, UNET_LAYER) 
    model.to(device) #先分配给gpu
    if pretrained_path is not None:
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        print(f'Loading initial_checkpoint: {pretrained_path}\n')
        checkpoint = torch.load(pretrained_path, device)
        model = load_model_checkpoint(model, checkpoint, device) 
    model = nn.DataParallel(model) 
    return model


def load_model_checkpoint(model, checkpoint, device):
    # 预加载自己训练的模型
    model.load_state_dict(checkpoint['state_dict']) #参数在cpu
    model.to(device) #再次分配到gpu
    return model
    

def load_optimizer_checkpoint(optimizer, checkpoint, device):
    # 继续训练会要记录上次optimizer的状况
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #参数在cpu
    #param_groups会跟随model.to(device)而转换到gpu，但是state仍然在cpu
    for state in optimizer.state.values(): #取出里面的mementum_buffer
        for k, v in state.items():
            state[k] = v.to(device) #再次分配到gpu
    return optimizer