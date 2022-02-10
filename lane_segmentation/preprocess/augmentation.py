import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from config import *
from mask import *
from image import *

def ImageAug(image, mask): #增强image
    if np.random.rand() > 0.5: #一半机会
        seq = iaa.Sequential([iaa.OneOf([ #任意一个
            iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
            iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
            iaa.GaussianBlur(sigma=(0, 1.0))])])
        image = seq.augment_image(image)
    return image, mask


def DeformAug(image, mask): #随机剪裁image和mask
    seq = iaa.Sequential([ #随机对w,h两边同时crop 0-5%或者pad 0-10%
        iaa.CropAndPad(percent=(-0.05, 0.1))]) #随机范围
    #默认是stochastic随机状态，转成deterministic相当于每个batch内固定random seed，用相同的随机值
    seg_to = seq.to_deterministic() 
    image = seg_to.augment_image(image)
    mask = seg_to.augment_image(mask)
    return image, mask


def ScaleAug(image, mask): #缩放image和mask
    scale = np.random.uniform(0.7, 1.5)
    h, w, _ = image.shape
    aug_image = image.copy()
    aug_mask = mask.copy()
    aug_image = cv2.resize(aug_image, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_LINEAR)
    aug_mask = cv2.resize(aug_mask, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_NEAREST)
    if (scale < 1.0): #如果缩小，用pad维持原图大小
        new_h, new_w, _ = aug_image.shape
        pre_h_pad = int((h - new_h) / 2)
        pre_w_pad = int((w - new_w) / 2)
        pad_list = [[pre_h_pad, h - new_h - pre_h_pad], [pre_w_pad, w - new_w - pre_w_pad], [0, 0]]
        aug_image = np.pad(aug_image, pad_list, mode="constant")
        aug_mask = np.pad(aug_mask, pad_list[:2], mode="constant")
    if (scale > 1.0): #如果放大，crop掉两边维持原图大小
        new_h, new_w, _ = aug_image.shape
        pre_h_crop = int ((new_h - h) / 2)
        pre_w_crop = int ((new_w - w) / 2)
        post_h_crop = h + pre_h_crop
        post_w_crop = w + pre_w_crop
        aug_image = aug_image[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
        aug_mask = aug_mask[pre_h_crop:post_h_crop, pre_w_crop:post_w_crop]
    return aug_image, aug_mask

    
def CutOut(image, mask): #随机挖空image，模拟遮挡
    patch_size=32 #遮挡的方块边长
    patch_size_half = patch_size // 2
    offset = 1 if patch_size % 2 == 0 else 0 #偶数边长，offset 1，因为randint不取后面值[)
    h, w = image.shape[:2] #假设为(100,100)
    #原图两边留一半的patch size为边缘，中间部分随意
    cxmin, cxmax = patch_size_half, w + offset - patch_size_half #(16,85)内
    cymin, cymax = patch_size_half, h + offset - patch_size_half
    cx = np.random.randint(cxmin, cxmax) #方块中心点的位置
    cy = np.random.randint(cymin, cymax)
    xmin, ymin = cx - patch_size_half, cy - patch_size_half #左上角
    xmax, ymax = xmin + patch_size, ymin + patch_size #右下角
    xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)
    
    if np.random.rand() > 0.5: #一半机会
        image[ymin:ymax, xmin:xmax] = (0, 0, 0) #用黑色方块遮挡
    return image, mask


def augumentor(image, mask):
    aug_func = [ImageAug, DeformAug, ScaleAug, CutOut]
    for aug in aug_func:
        image, mask = aug(image, mask)
    return image, mask    


def show_aug(data, aug_func):
    image = plt.imread(data['images'][0])
    mask = plt.imread(data['labels'][0]) #33个类别 [0,1] 
    mask = (mask*255).astype('uint8')
    mask = convert_labels(mask) #8个类别
    image, mask = crop_and_resize(image, mask)
    image, mask = aug_func(image, mask)
    images = [image,mask]
    plt.figure(figsize=(15,4))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(images[i], cmap='gray') 
        plt.title(f'{images[i].shape}, {images[i].dtype}')
        plt.axis('off')
    plt.show()


if __name__=='__main__':
	train = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
	show_aug(train, CutOut)