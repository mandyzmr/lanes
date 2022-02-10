import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import *

# 根据路径创建一一对应的data list，路径是从road->record->camera，这里只要灰色mask的路径
def create_data_list(path, train_size=0.8):
    image_path_pattern = os.path.join(path, 'Road**','ColorImage_road**','**','**.jpg') #遍历彩图下所有jpg文件
    color_mask_path_pattern = os.path.join(path, 'Road**','Labels_road**','**','**_bin.png') #遍历彩mask下所有png文件
    mask_path_pattern = os.path.join(path, 'Gray_Label','**','**_bin.png') #遍历灰mask下所有png文件
    image_paths = sorted(glob.glob(image_path_pattern,recursive=True)) #路径按顺序排序
    color_mask_paths = sorted(glob.glob(color_mask_path_pattern,recursive=True))
    mask_paths = sorted(glob.glob(mask_path_pattern,recursive=True))

    train_img, test_img, train_color_mask, test_color_mask, train_mask, test_mask  = train_test_split(image_paths, color_mask_paths, mask_paths, train_size=train_size, shuffle=True)
    train = pd.DataFrame({'images':train_img, 'color_labels': train_color_mask, 'labels':train_mask})
    test = pd.DataFrame({'images':test_img, 'color_labels': test_color_mask, 'labels':test_mask})
    train.to_csv(os.path.join(DATA_ROOT, 'train.csv'), index=False)
    test.to_csv(os.path.join(DATA_ROOT, 'test.csv'), index=False)
    print(f'We have created {len(train)} samples for training, and {len(test)} samples for validation.')
    return train, test


# 随机查看样本
def show_sample(data, idx): #显示图片/mask
    # 随机查看样本
    image = plt.imread(data['images'][idx])
    color_mask = plt.imread(data['color_labels'][idx]) #混合单通道和4通道图
    gray_mask = plt.imread(data['labels'][idx]) #33个类别 [0,1] 
    images = [image, color_mask, gray_mask]
    plt.figure(figsize=(15,8))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i], cmap='gray') 
        plt.title(f'{images[i].shape}, {images[i].dtype}')
        plt.axis('off')
    plt.show()


if __name__=='__main__':
	train, val = create_data_list(os.path.join(DATA_ROOT, TRAIN_PATH))
	print(train.head())
	# idx = np.random.choice(train.shape[0])
	# show_sample(train, idx)