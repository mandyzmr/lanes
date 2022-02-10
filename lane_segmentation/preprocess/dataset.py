import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from config import *
from create_lmdb import *
from image import *
from augmentation import *


class LaneDataset(Dataset):
    def __init__(self, transform=None, mode='train'):
        super(LaneDataset, self).__init__()
        self.data = pd.read_csv(os.path.join(DATA_ROOT, f'{mode}.csv'), usecols=['images','labels'])
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 读取
        image_path = self.data['images'][idx]
        mask_path = self.data['labels'][idx]
        # image, mask = LMDB2image(image_path, mask_path, self.mode) #(h,w,3),(h,w), uint8
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 处理
        image, mask = crop_and_resize(image, mask) #剪裁天空
        mask = convert_labels(mask) #转成8个类别
        if self.mode=='train':
            image, mask = augumentor(image, mask) #数据增强
        if self.transform:
            image = self.transform(image) #float32, [0,1]
        mask = torch.LongTensor(mask) #int64
        return image, mask


def show_sample(sample):
    image = np.transpose(sample[0].numpy(), (1,2,0))[...,::-1] #(h,w,c), rgb
    mask = sample[1].numpy()
    mask = mask2color(mask) #用彩色查看
    images = [image, mask]
    plt.figure(figsize=(15,4))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(images[i], cmap='gray') 
        plt.title(f'{images[i].shape}, {images[i].dtype}')
        plt.axis('off')
    plt.show()


if __name__=='__main__':
    # train_dataset = LaneDataset(transform=train_transform, mode='train')
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, 
    #                           drop_last=True, num_workers=0)
    valid_dataset = LaneDataset(transform=test_transform, mode='test')
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE//2, 
                              drop_last=False, num_workers=0)
    from tqdm import tqdm
    for i in tqdm(valid_loader):
        pass
    
    # idx = np.random.choice(train_dataset.__len__())
	# sample = train_dataset[idx]
	# show_sample(sample)
    
