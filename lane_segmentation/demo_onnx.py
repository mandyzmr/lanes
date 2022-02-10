import os
import torch
from torch import nn
from torch import onnx
from preprocess.image import *
from lane_segmentation import *
from config import *


def torch2onnx(name):
    # 加载模型
    lane_seg = LaneSegmentation(name)
    model = lane_seg.model 
    
    # 模拟输入
    dummy_image  = np.random.randint(0,256,[IMAGE_SIZE[1],IMAGE_SIZE[0],3], dtype=np.uint8) # 模型输入维度为(1, 3, 224, 224)
    dummy_image = crop_and_resize(dummy_image)
    dummy_image = test_transform(dummy_image)[np.newaxis,...] #[0,1] (1,c,h,w)
    dummy_image = dummy_image.to(device)
    
    # 导出模型为ONNX格式，通过运行一次模型获得模型的执行轨迹
    save_path = os.path.splitext(lane_seg.pretrained_path)[0]+'.onnx'
    onnx.export(model, dummy_image, save_path, opset_version=11, verbose=True, 
                input_names=['image'], output_names=['mask'])


if __name__ == '__main__':
    torch2onnx('deeplabv3p')