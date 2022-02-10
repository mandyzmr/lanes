import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import *
from tqdm import tqdm


# 根据8个类别的对应原label列表，把mask的原label转为新label
def convert_labels(mask): 
    encode_mask = np.zeros(mask.shape, dtype='uint8') #(h,w)
    id_train = {0:[0, 249, 255, 213, 206, 207, 211, 208,216,215,218, 219,232, 202, 231,230,228,229,233,212,223],
                1:[200, 204, 209], 2: [201,203], 3:[217], 4:[210], 5:[214],
                6:[220,221,222,224,225,226], 7:[205,227,250]}
    for i in range(8):
        for label in id_train[i]:
            encode_mask[mask == label] = i
    return encode_mask


# 可视化mask
def mask2gray(mask): #用灰度图显示mask
    color_map = {0:0, 1:204, 2:203, 3:217, 4:210, 5:214, 6:224, 7:227}
    gray_mask = np.zeros(mask.shape, dtype='uint8')
    
    for i in range(8):
        gray_mask[mask == i] = color_map[i]
    return gray_mask


def mask2color(mask): #用RGB显示mask
    # 统一用原label里第一个color map作为每个新label的颜色
    color_map = {0:(0,0,0), 1:(70,130,180), 2:(0,0,142), 3:(153,153,153),
                 4:(128,64,128), 5:(190,153,153), 6:(0,0,230), 7:(255,128,0)}
    color_mask = np.zeros((mask.shape[0], mask.shape[1],3), dtype='uint8')
    
    for c in range(3):
        for i in range(8):
            color_mask[...,c][mask == i] = color_map[i][c]
    return color_mask


def show_mask(data, idx):
    label_path = data['labels'][idx]
    mask = plt.imread(label_path) #33个类别 [0,1] 
    mask = (mask*255).astype('uint8') #[0,255]
    en_mask = convert_labels(mask) #8个类别
    gray_mask = mask2gray(en_mask)
    color_mask = mask2color(en_mask)
    masks = [mask, en_mask, gray_mask, color_mask]
    titles = ['Original mask','Training mask','Gray Visualiazation','Color Visualization']
    
    plt.figure(figsize=(15,8))
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.imshow(masks[i],cmap='gray') 
        plt.title(titles[i])
        plt.axis('off')
    # plt.show()


def check_labels(data, num, mode='train'): #随机查看mask的label分布情况
    labels = np.zeros(8)
    idx = np.random.choice(data.shape[0], num, replace=False)
    for i in tqdm(idx): 
        mask = plt.imread(data['labels'][i]) #33个类别 [0,1] 
        mask = (mask*255).astype('uint8')
        mask = convert_labels(mask) #8个类别
        for i in range(8):
            labels[i]+=np.sum(mask==i)

    #忽略最多的label 0，按频数可视化
    plt.bar(np.arange(1,8), labels[1:]) #忽略最多的label 0
    plt.title(f'{mode} set @ {num} samples'.capitalize())
    plt.savefig(f'{mode}_distribution.jpg')
    plt.show()


def poly2mask(ann_path):
    with open(ann_path, 'r') as f:
        ann = json.load(f)
        
    # 创建黑底的annot mask
    h, w = ann['imageHeight'], ann['imageWidth'] #原图大小
    mask = np.zeros((h, w), dtype=np.uint8) 
    
    # 标注多边形
    polys = defaultdict(list) #多边形列表，每个元素为一个多边形的端点
    i = 0
    for poly_ann in ann['shapes']:
        if poly_ann['shape_type'] != 'polygon': #只画多边形
            continue
        poly = np.array(poly_ann['points']).astype(np.int64) #把点转为int
        polys[poly_ann['label']].append(poly) #放到不同label下
        i+=1
        
    # 填充颜色
    color_map = {'left':100, 'straight':255}
    for label in polys:
        color = color_map[label] #不同label标注不同颜色
        cv2.fillPoly(mask, polys[label], color) #用255白色填充多个多边形
    
    # 保存mask
    save_path = os.path.splitext(ann_path)[0]+'.png' #xxx.png
    cv2.imwrite(save_path, mask)
    print(f"There are {i} lanes detected.")
    return mask

    
if __name__=='__main__':
    train = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
    # idx = np.random.choice(train.shape[0])
    # show_mask(train, idx)
    
    check_labels(train, 1000, 'train')


