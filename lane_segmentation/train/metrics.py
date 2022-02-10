import glob
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from config import *


def calculate_mean_iou(y_pred, y_true, epsilon=1e-3):
    iou = {'inter':0.,'union':0.} #不需要计算梯度，所以tensor先detach，但是为了速度，继续在gpu上运算
    y_true = F.one_hot(y_true, NUM_CLASSES).permute(0,3,1,2).detach() #转one hot (b,h,w)->(b,c,h,w)
    y_pred = F.one_hot(torch.argmax(y_pred, dim=1), NUM_CLASSES).permute(0,3,1,2).detach()  #logits转为label，再转为one hot
    inter = torch.sum(y_pred*y_true, dim=[0,2,3]) #每个类别预测正确总数，int
    union = torch.sum(y_pred+y_true, dim=[0,2,3]) - inter
    iou['inter'] += inter #每个类别的情况，float
    iou['union'] += union
    iou['union'][iou['union']==0.]=epsilon #用epsilon替换union为0时的值
    mean_iou = torch.mean(iou['inter']/iou['union']).item() #得到mean iou
    return mean_iou, iou


def plot_iou(iou_rate, epoch, iou_dir=''):
    plt.bar(range(NUM_CLASSES), iou_rate)
    plt.title(f'IOU (Epoch {epoch:02})')
    plt.xlabel('classes', fontsize=12)
    margin = np.max(iou_rate)*0.01 #按比例把数字标记在bar上
    for x,y in zip(range(NUM_CLASSES), iou_rate):
        plt.text(x-0.3, y+margin, f'{y:.2%}')
    plt.savefig(os.path.join(iou_dir, f'iou_epoch{epoch:02}.jpg'))
    # plt.show();
    plt.close() #防止后续图片重叠


def generate_iou_gif(iou_dir=''):
    filenames = glob.glob(os.path.join(iou_dir,'iou_epoch*')) #完整路径，同时避免.DS_stores等隐藏文件
    filenames = sorted(filenames) #确保按顺序
    plots = []
    for filename in filenames:
        plots.append(plt.imread(filename))
    imageio.mimsave(os.path.join(iou_dir, 'iou.gif'), plots, 'GIF-FI', fps=2)
    # display(IPyImage(open(os.path.join(iou_dir, 'iou.gif'), 'rb').read())) #显示动画
    