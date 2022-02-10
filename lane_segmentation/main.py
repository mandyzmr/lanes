import os
import pickle
from collections import defaultdict
import time
import argparse
import sys
sys.path.append('./preprocess') #增加相对路径
sys.path.append('./model') 
sys.path.append('./train')
# sys.path.append('/home/aistudio/external_library') #ai studio应用环境
from utils import *
from summary import *
from config import *
from preprocess.dataset import *
from train.train_infer import *
from train.optimizer import *
from train.criterion import *


def run_train(config):
    # ------setup------
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_{config.backbone}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    iou_dir = os.path.join(out_dir, 'iou_plots')
    for directory in [checkpoint_dir, iou_dir]:
        if not os.path.exists(directory): 
            os.makedirs(directory) #多层创建文件夹
   
    log = Logger()
    log.open(os.path.join(out_dir, f'{model_name}_training.txt'), mode='a') #增加内容
    log.write(f'Training log @ {out_dir}\n')
    log.write(f'Device: {device}\n')
    log.write('\n')

    #------dataset------
    log.write('** Dataset setting **\n')
    train_dataset = LaneDataset(transform=train_transform, mode='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, 
                              drop_last=True, num_workers=num_workers, pin_memory=pin_memory)
    valid_dataset = LaneDataset(transform=test_transform, mode='test')
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE, 
                              drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
    log.write(f'training_samples = {train_dataset.__len__()}\n')
    log.write(f'val_samples = {valid_dataset.__len__()}\n')
    log.write(f'batch_size = {BATCH_SIZE}\n')
    log.write('\n')
    
    #------model------
    log.write('** Model setting **\n')
    model = get_model(config.model, config.backbone, OUTPUT_STRIDE, DILATION, PRETRAINED, UNET_LAYER)
    log.write(f'Model: {type(model).__name__} ({config.backbone})\n')
    if config.model=='deeplabv3p':
        log.write(f'os = {OUTPUT_STRIDE}\n')
        log.write(f'dilation = {DILATION}\n')
    elif config.model=='unetpp':
        log.write(f'unet_layer = {UNET_LAYER}\n')
    model.to(device) #先分配给gpu
    log.write('\n')
    
    #------train------
    log.write('** Training setting **\n')
    optimizer = get_optimizer(model, name=OPTIMIZER)
    log.write(f'criterion = 10*focal_loss and dice_loss\n')
    log.write(f'optimizer = {type(optimizer).__name__}\n')
    log.write(f'epochs = {config.epochs}\n')
    log.write('\n')

    #------pretrained------
    epoch_start = 0
    max_mean_iou = 0.0
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        log.write(f'Loading initial_checkpoint: {pretrained_path}\n')
        checkpoint = torch.load(pretrained_path, device)
        model = load_model_checkpoint(model, checkpoint, device) #加载完之后，再用DP
        # 当训练意外中断时，加载方便继续训练
        # optimizer = load_optimizer_checkpoint(optimizer, checkpoint, device)
        epoch_start = checkpoint['epoch']
        max_mean_iou = checkpoint['mean_iou']
        log.write(f'Max mean iou: {max_mean_iou:.4f}')
        log.write('\n')
    model = nn.DataParallel(model) #gpu多于1个时，并行运算

    #------log------
    log.write('** Start training here! **\n')
    pattern1="{: ^12}|{:-^33}|{:-^79}|\n" #开头第一行 
    pattern2="{: ^6}"*2+"|"+"{: ^11}"*3+"|"+"{: ^11}"*3+"{: ^46}"+"|"+"{: ^12}\n" #标题行
    pattern3="{: ^6}"+"{: ^6.0e}"+"|"+"{: ^11.4f}"*3+"|"+"{: ^11.4f}"*3+"{: ^46}"+"|"+"{: ^12}\n" #内容行
    log.write(pattern1.format('',' TRAIN ',' VALID '))
    log.write(pattern2.format('epoch','lr','focal_loss','dice_loss', 'mean_iou','focal_loss','dice_loss', 'mean_iou','iou', 'time'))
    log.write("-"*136+'\n')       
    
    history = defaultdict(list)
    val_focal_loss, val_dice_loss, val_mean_iou = 0,0,0 #前半周期不做validation
    val_iou = np.zeros(8)
    start = time.time() #计时
    for e in range(epoch_start, epoch_start+config.epochs): #继续从上次的epoch训练 
        print(f'Epoch {e+1}/{epoch_start+config.epochs}')
        # 根据epoch先调整lr
        lr = adjust_learning_rate(optimizer, e)

        # 训练
        train_focal_loss, train_dice_loss, train_mean_iou = train_on_epoch(model, train_loader, optimizer, device, e, iou_dir)
        history['train_focal_loss'].append(train_focal_loss)
        history['train_dice_loss'].append(train_dice_loss)
        history['train_mean_iou'].append(train_mean_iou)
        
        # 验证
        if valid_loader: 
            val_focal_loss, val_dice_loss, val_mean_iou, val_iou = val_on_epoch(model, valid_loader, device, e, iou_dir)
            history['val_focal_loss'].append(val_focal_loss)
            history['val_dice_loss'].append(val_dice_loss)
            history['val_mean_iou'].append(val_mean_iou)
            history['val_iou'].append(val_iou)
            
            end = time.time() #每个epoch结束后计算一次累计时间
            log.write(pattern3.format(e+1, lr, train_focal_loss, train_dice_loss, train_mean_iou, val_focal_loss, val_dice_loss, val_mean_iou, 
                                      ', '.join([f'{i:.2f}'for i in val_iou]), time_to_str(end - start)))
            
            if val_mean_iou > max_mean_iou:
                max_mean_iou = val_mean_iou #更新最大mean iou值
                ckpt_path = os.path.join(checkpoint_dir, f'global_max_mean_iou_model.pth') #仅保存一个最优模型
                torch.save({
                    'epoch':e+1, 
                    'mean_iou': max_mean_iou,
                    'model':type(model.module).__name__,
                    'state_dict': model.module.state_dict(), #模型参数w/b信息
                    'optimizer_state_dict': optimizer.state_dict(), #包括bn的running mean和std等信息
                }, ckpt_path)
                log.write(f'Saving epoch {e+1} max mean iou model: {max_mean_iou:.4f}\n')
        else:
            end = time.time() #每个epoch结束后计算一次累计时间
            log.write(pattern3.format(e+1, lr, train_focal_loss, train_dice_loss, train_mean_iou, val_focal_loss, val_dice_loss, val_mean_iou, 
                                      ', '.join([f'{i:.2f}'for i in val_iou]), time_to_str(end - start)))
            
    # 可视化iou变化情况
    generate_iou_gif(iou_dir)
    # 保存每个epoch的metrics结果，方便后续可视化查看训练情况
    pickle.dump(history, open(os.path.join(out_dir, f'{model_name}_history.pkl'),'wb'))


def run_test(config):
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_{config.backbone}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
         
    model = get_model(config.model, config.backbone, OUTPUT_STRIDE, DILATION, PRETRAINED, UNET_LAYER)
    model.to(device) #先分配给gpu
    
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        print(f'Loading initial_checkpoint: {pretrained_path}\n')
        checkpoint = torch.load(pretrained_path, device)
        model = load_model_checkpoint(model, checkpoint, device) 
    model = nn.DataParallel(model) 
    
    image_path = os.path.join(DATA_ROOT, TEST_PATH, config.image_path)
    image = cv2.imread(image_path)
    mask = predict(model, image, device, config.color_mode) #得到真人的概率
    
    print('Rendering...')
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.imshow(image[...,::-1]) 
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray') 
    
    if config.save:
        save_path = os.path.join(DATA_ROOT, PRED_PATH, os.path.splitext(config.image_path)[0]+f'{"_color" if config.color_mode else ""}_mask.jpg')
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR) #把mask从RGB转为BGR
        cv2.imwrite(save_path, mask)
        print('Prediction mask saved.')
    return mask


def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer':
        return run_test(config)


if __name__ == '__main__':
    # 在终端传入参数运行模型
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--model', type=str, default='deeplabv3p', choices=['deeplabv3p','unet','unetpp'])
    parser.add_argument('--backbone', type=str, default='', choices=['resnet18','resnet34','resnet50','resnet101','resnet152'])
    
    # train
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train', choices=['train','infer'])
    parser.add_argument('--pretrained_model', type=str, default=None) #预训练模型路径 global_min_acer_model.pth
    
    # test
    parser.add_argument('--image_path', type=str, default=None) #预测图片路径
    parser.add_argument('--color_mode', type=bool, default=False) #预测图片路径
    parser.add_argument('--save', type=bool, default=True) #保存预测mask
    
    config = parser.parse_args()
    print(config)
    main(config)

