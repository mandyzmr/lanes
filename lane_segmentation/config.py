import torch
from torchvision import transforms 
import multiprocessing
import sys
import os

'''
此处定义常用固定配置，若需要根据情况训练不同模型，直接通过main.py用终端指定参数进行训练和预测。
'''

#------Device------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_cores = multiprocessing.cpu_count()
torch.backends.cudnn.benchmark=True #cuDNN autotuner

#------Original Dataset------
DATA_ROOT = 'dataset' 
TRAIN_PATH = 'Train'
TEST_PATH = 'Test'
PRED_PATH = 'Prediction'
LMDB_PATH = 'LMDB'
TRAIN_LIST = 'train.csv' 
VAL_LIST = 'test.csv'
     
#------Dataset Loader------
OFFSET = 690 #裁掉原图h上半部分天空的长度
IMAGE_SIZE = (3384, 1710) #包括天空的原图大小 
INPUT_SIZE = (1024, 384) #剪裁后输入模型的图片大小，还可以尝试(1536,512)
BATCH_SIZE = 6 #deeplabv3p用6-8，unetpp用2
train_transform = transforms.Compose([
    transforms.ToTensor(), #把array/PIL从[0,255]转为[0,1]的FloatTensor [c,h,w]
    # transforms.RandomHorizontalFlip(), # 随机翻转 PIL/tensor，同时输出mask的时候不用random flip
    # transforms.Normalize( #对tensor的channel进行normalization
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(), #把array/PIL从[0,255]转为[0,1]的FloatTensor [c,h,w]
    # transforms.Normalize(
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5])
])

if device.__str__() == 'cuda':
    num_workers = 4 #deeplabv3p用4，unetpp用3
    pin_memory = True #使用锁页内存加速复制数据到gpu
else: #AIStudio终端查看df -h可以看到在CPU配置下，shm只有64m，需要增加limit防止爆内存docker run--shm-size 8g，若没有权限，需要设置num_workers=0
    num_workers = 0
    pin_memory = False

#------Model------
OUTPUT_STRIDE = 16
DILATION = [1,2,4] #resnet block5的multi-scale dilation
ASPP_DIM = 256
SHORTCUT_DIM = 48
SHORTCUT_KERNEL = 1
UNET_LAYER = 4 #1,2,3,4
NUM_CLASSES = 8
PRETRAINED = False

#------Train------
OPTIMIZER = 'sgd' #sgd, adam
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.0e-4
CRITERION = 'focal_loss' #cross_entropy, dice_loss, focal_loss
# WEIGHT = (torch.tensor([2.0532e-04, 1.5655e-02, 6.7465e-02, 1.7170e-01,
#                        4.9341e-01, 1.8606e-02, 5.2471e-02, 1.8049e-01])*8).to(device)
WEIGHT = torch.ones(8).to(device) #相等类别权重

