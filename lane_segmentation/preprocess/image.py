import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from config import *
from mask import *


# 剪裁原图，把图片上半部分亮色天空去掉，免得影响识别白色车道线，再resize：(w,h)->(w,h‘)->(tw,th)
def crop_and_resize(image, mask=None):
    roi_image = image[OFFSET:, :] #只取图片下半部分
    if mask is not None:
        roi_mask = mask[OFFSET:, :] #label同样处理
        image = cv2.resize(roi_image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(roi_mask, INPUT_SIZE, interpolation=cv2.INTER_NEAREST) #因为label是特定数字，不能用其他计算的方式
        return image, mask
    else:
        image = cv2.resize(roi_image, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        return image


# 再把预测mask重新恢复原图大小：(tw,th)->(w,h')->(w,h)
def expand_resize_mask(mask, size=IMAGE_SIZE, offset=OFFSET):
    pred_mask = mask2gray(mask) 
    expand_mask = cv2.resize(pred_mask, (size[0], size[1]-offset), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((size[1], size[0]), dtype='uint8')
    mask[offset:, :] = expand_mask #还原原图裁掉上半部分，并默认为黑色，下半部分拼接
    return mask


def expand_resize_color_mask(mask, size=IMAGE_SIZE, offset=OFFSET):
    color_pred_mask = mask2color(mask) #得到RGB的array图
    color_expand_mask = cv2.resize(color_pred_mask, (size[0], size[1] - offset), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros((size[1], size[0], 3), dtype='uint8')
    color_mask[offset:, :, :] = color_expand_mask
    return color_mask


# 随机查看resize后的样本
def show_sample(data, idx, color_mode=True): 
    image = plt.imread(data['images'][idx])
    mask = plt.imread(data['labels'][idx]) #33个类别 [0,1] 
    mask = (mask*255).astype('uint8')
    mask = convert_labels(mask) #8个类别
    resize_image, resize_mask = crop_and_resize(image, mask) 
    if color_mode:
        expand_mask = expand_resize_color_mask(resize_mask)
    else:
        expand_mask = expand_resize_mask(resize_mask)
    images = [resize_image, resize_mask, image, expand_mask]
    
    plt.figure(figsize=(15,8))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i], cmap='gray') 
        plt.title(f'{images[i].shape}, {images[i].dtype}')
        plt.axis('off')


# 把gray mask转化为expand mask，即预测mask的true mask
def get_true_mask(path, color_mode=True): 
    mask = plt.imread(path) #33个类别 [0,1] 
    mask = (mask*255).astype('uint8')
    mask = convert_labels(mask) #8个类别
    image = np.random.rand(mask.shape+(3,)) #fake
    resize_image, resize_mask = crop_and_resize(image, mask) 
    if color_mode:
        expand_mask = expand_resize_color_mask(resize_mask) #rgb
    else:
        expand_mask = expand_resize_mask(resize_mask)
    expand_mask = cv2.cvtColor(expand_mask, cv2.COLOR_RGB2BGR) #把mask从RGB转为BGR
    cv2.imwrite('converted_'+path, expand_mask)



if __name__=='__main__':
	# train = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
	# idx = np.random.choice(train.shape[0])
	# show_sample(train, idx)
    get_true_mask('dataset/Test/170927_064448626_Camera_6_bin.png')
