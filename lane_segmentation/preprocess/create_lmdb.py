import lmdb
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from config import *
from tqdm import tqdm


#将image和label图写入LMDB结构中，key{image,label}，value{图片的二进制值}
def path2LMDB(data, mode='train'):
    if not os.path.exists(os.path.join(DATA_ROOT, LMDB_PATH, mode)): #先判断是否存在文件夹，再新建，否则覆盖
        os.makedirs(os.path.join(DATA_ROOT, LMDB_PATH, mode)) 
    
    #创建LMDB文件的环境路径，或者用with lmdb.open as env: 此处因为都写入同个path就不重复初始化环境
    #map_size定义最大储存容量，单位是b，以下定义1TB容量2**40b，否则容易爆limit
    env = lmdb.open(os.path.join(DATA_ROOT, LMDB_PATH, mode), map_size=2**40)
    
    # 方法一：数据少时，直接df.apply得到二进制，然后cache=dict(zip(key,value))，但是空间复杂度高
    # 方法二：数据多时，为了防止爆内存，逐个样本cache，但是时间复杂度高
    for i in tqdm(range(data.shape[0])):
        cache = {}  
        with open(data['images'][i],'rb') as f:
            value = f.read() #得到二进制图片
            cache[data['images'][i].encode()] = value #把路径作为key，不容易混乱，也转为二进制utf8字符串
 
        with open(data['labels'][i],'rb') as f:
            value = f.read() 
            cache[data['labels'][i].encode()] = value 
        
        #写入文件，或者用txn=env.begin，结束后txn.commit()
        with env.begin(write=True) as txn: 
            for k, v in cache.items():
                txn.put(k, v) #写入/更新字典，还有其他常用指令如txn.delete(k),txn.cursor()类似dict.items()
    env.close() #如果写成with格式，就不需要


#从lmdb文件中读取图片
def LMDB2image(image_path, label_path, mode):
    env = lmdb.open(os.path.join(DATA_ROOT, LMDB_PATH, mode))
    with env.begin(write=False) as txn: #读取非写入
        # 读取字典，得到二进制图片和label mask
        image_bin = txn.get(image_path.encode())
        label_bin = txn.get(label_path.encode()) 

        # 将数组解码成(h,w,c)的(0,255)的uint8数组，label从float转为uint8
        image = np.array(bytearray(image_bin), dtype=np.uint8) #二进制->一维数组
        label = np.array(bytearray(label_bin), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) #(h,w,3)
        label = cv2.imdecode(label, cv2.IMREAD_GRAYSCALE) #(h,w)
    env.close()
    return image, label


def show_sample(data, idx, mode='train'):
    image_path = data['images'][idx]
    label_path = data['labels'][idx]
    image, mask = LMDB2image(image_path, label_path, mode)
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.imshow(image[...,::-1]) 
    plt.title(f'{image.shape}, {image.dtype}')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mask,cmap='gray') 
    plt.title(f'{mask.shape}, {mask.dtype}');
    plt.axis('off')
    plt.show()


if __name__=='__main__':
	# 得到两个mdb文件
	train = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
	test = pd.read_csv(os.path.join(DATA_ROOT, 'test.csv'))
	path2LMDB(train, 'train')
	# path2LMDB(test, 'test')

	idx = np.random.choice(train.shape[0])
	show_sample(train, idx, 'train')

