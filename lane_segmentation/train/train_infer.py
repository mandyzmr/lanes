import time
import torch
import numpy as np
from metrics import *
from preprocess.image import *
from preprocess.mask import *
from train.criterion import *
from config import *


def train_on_epoch(model, train_loader, optimizer, device, epoch, iou_dir, iter_smooth=100):
    epoch_ce_loss = 0
    # epoch_focal_loss = 0
    epoch_dice_loss = 0
    epoch_iou = {'inter':0.,'union':0.} #由于每个batch的pixels数不固定，需要累积求epoch均值
    n_batches = len(train_loader)
    # start = time.time()
    for i, (image, mask) in enumerate(train_loader):
        model.train() #训练模式
        image = image.to(device) #(b,c,h,w)
        mask = mask.to(device) #(b,h,w)
        # print(f'Loading time: {time.time()-start}')
        
        # start = time.time()
        # 计算分割结果和iou
        y_pred = model(image) #(b,n_classes,h,w)
        ce_loss = F.cross_entropy(y_pred, mask, WEIGHT)
        # focal_loss = FocalLoss()(y_pred, mask, WEIGHT)
        dice_loss = DiceLoss()(y_pred, mask, WEIGHT)
        loss = ce_loss+dice_loss
        # loss = 20*focal_loss+dice_loss
        mean_iou, iou = calculate_mean_iou(y_pred, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(f'Processing time: {time.time()-start}')
        epoch_ce_loss += ce_loss.item()
        # epoch_focal_loss += focal_loss.item()
        epoch_dice_loss += dice_loss.item()
        epoch_iou['inter']+=iou['inter']#.detach()
        epoch_iou['union']+=iou['union']#.detach()
        if i%iter_smooth==0:
            print(f'Batch {i+1}/{n_batches} - ce_loss: {epoch_ce_loss/(i+1):.3f} - dice_loss: {epoch_dice_loss/(i+1):.3f} - mean_iou: {mean_iou:.3f}') 
            # print(f'Batch {i+1}/{n_batches} - focal_loss: {epoch_focal_loss/(i+1):.3f} - dice_loss: {epoch_dice_loss/(i+1):.3f} - mean_iou: {mean_iou:.3f}') 
        # start = time.time()
    

    #仅在drop last的时候有效，否则求loss时reduction=‘none'，再除以样本总数
    epoch_ce_loss /= n_batches
    # epoch_focal_loss /= n_batches 
    epoch_dice_loss /= n_batches 
    epoch_iou = (epoch_iou['inter']/epoch_iou['union']).cpu().numpy() 
    epoch_mean_iou = np.mean(epoch_iou)
    return epoch_ce_loss, epoch_dice_loss, epoch_mean_iou
    # return epoch_focal_loss, epoch_dice_loss, epoch_mean_iou
    

def val_on_epoch(model, valid_loader, device, epoch, iou_dir):
    val_ce_loss = 0
    # val_focal_loss = 0
    val_dice_loss = 0
    val_iou = {'inter':0.,'union':0.}
    n_samples = 0
    for i, (image, mask) in enumerate(valid_loader):
        model.eval() #预测模式，关闭dropout和bn
        image = image.to(device)
        mask = mask.to(device)
        n_samples += image.shape[0]
        with torch.no_grad():
            y_pred = model(image) 
            #由于val batch没有drop last，先求样本loss总和，再求样本loss均值
            ce_loss = F.cross_entropy(y_pred, mask, WEIGHT, reduction='sum')
            # focal_loss = FocalLoss()(y_pred, mask, WEIGHT, reduction='sum')
            dice_loss = DiceLoss()(y_pred, mask, WEIGHT, reduction='sum')
            val_ce_loss += ce_loss.item()
            # val_focal_loss += focal_loss.item() #样本loss总和
            val_dice_loss += dice_loss.item()
            mean_iou, iou = calculate_mean_iou(y_pred, mask)
        
            # print(f'Processing time: {time.time()-start}')
            val_iou['inter']+=iou['inter']#.detach().cpu()
            val_iou['union']+=iou['union']#.detach().cpu()
    
    b,c,h,w = image.shape
    val_ce_loss = val_ce_loss/(n_samples*h*w)
    # val_focal_loss = val_focal_loss/(n_samples*h*w)
    val_dice_loss = val_dice_loss/(n_samples*NUM_CLASSES)
    val_iou = (val_iou['inter']/val_iou['union']).cpu().numpy() #每个类别的iou
    val_mean_iou = np.mean(val_iou)
    plot_iou(val_iou, epoch+1, iou_dir)
    print(f"Validation - ce_loss: {val_ce_loss:.3f} - dice_loss: {val_dice_loss:.3f} - mean_iou: {val_mean_iou:.3f} - iou: {', '.join([f'{i:.2f}'for i in val_iou])}")   
    # print(f"Validation - focal_loss: {val_focal_loss:.3f} - dice_loss: {val_dice_loss:.3f} - mean_iou: {val_mean_iou:.3f} - iou: {', '.join([f'{i:.2f}'for i in val_iou])}")   
    return val_ce_loss, val_dice_loss, val_mean_iou, val_iou
    # return val_focal_loss, val_dice_loss, val_mean_iou, val_iou
    


def predict(model, image, device, color_mode=True): #根据face_align预测emb
    model.eval() #[0,255] (h,w,c)
    image = crop_and_resize(image)
    image = test_transform(image)[np.newaxis,...] #[0,1] (1,c,h,w)
    image = image.to(device)
    with torch.no_grad():
        y_pred = model(image) #(1,c,h,w)
        y_pred = torch.argmax(y_pred.detach().cpu(), dim=1).squeeze(0) #(h,w)
        y_pred = y_pred.numpy()
        if color_mode:
            mask = expand_resize_color_mask(y_pred)
        else:
            mask = expand_resize_mask(y_pred)
    return mask