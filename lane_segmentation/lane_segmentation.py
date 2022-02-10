import os
import torch
import sys
sys.path.append('./model') #增加绝对路径
sys.path.append('./preprocess')
sys.path.append('./train')
from summary import *
from config import *
from train.train_infer import *


class LaneSegmentation():
    def __init__(self, name='deeplabv3p'):
        super(LaneSegmentation, self).__init__()
        # 加载预训练模型
        pretrained_models = {'deeplabv3p': {'setting':('deeplabv3p','resnet101',16,[1,2,4],False,4),
                                           'checkpoint': os.path.join('running_log','deeplabv3p_resnet101', 'checkpoint', 'global_max_mean_iou_model.pth')},
                             'unetpp': {'setting':('unetpp','resnet50',16,[1,2,4],False,4),
                                        'checkpoint': os.path.join('running_log','unetpp_resnet50', 'checkpoint', 'global_max_mean_iou_model.pth')}}
        self.model = get_model(*pretrained_models[name]['setting'])
        self.pretrained_path = pretrained_models[name]['checkpoint'] #为了后续可以转变成其他模型格式时，保存使用
        checkpoint = torch.load(self.pretrained_path, device)
        self.model = load_model_checkpoint(self.model, checkpoint, device) 
        self.model.to(device) 
      
    def get_mask(self, image, color_mode=True): 
        mask = predict(self.model, image, device, color_mode) 
        return mask


if __name__=='__main__':
    # 调用车道线分割模型
    lane_seg = LaneSegmentation()
    image = cv2.imread('dataset/Test/test.jpg')
    mask = lane_seg.get_mask(image)
    plt.imshow(mask)
    plt.axis('off')
    plt.show()