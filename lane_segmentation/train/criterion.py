import torch
from torch import nn
import torch.nn.functional as F
from config import *


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma #增加困难样本比重
        
    #为了和F.cross_entropy调用方式统一，在运算过程中决定reduction方式和weight
    def forward(self, y_pred, y_true, alpha=None, reduction='mean'):
        if alpha is None: #类别比重
            alpha = torch.ones(NUM_CLASSES).to(device) #保证所有运算在gpu上
        alpha = alpha[y_true] #(b,h,w) 得到每个像素对应类别的权重
        
        y_pred = F.softmax(y_pred, dim=1) #一旦detach就没有gradfunc
        y_true = F.one_hot(y_true, NUM_CLASSES).permute(0,3,1,2) #(b,h,w)->(b,c,h,w)
        y_pred = torch.sum(y_pred*y_true, dim=1) #(b,h,w) 只剩下正确类别的概率
        y_pred = y_pred.clamp(min=1e-4,max=1.0) #log(1e-45)为止还有值，之后就变-inf了
        focal_loss = -alpha * (1.-y_pred).pow(self.gamma) * torch.log(y_pred) 
        if reduction == 'mean': 
            return focal_loss.mean() #求pixel均值，大概0.014左右
        elif reduction == 'sum':
            return focal_loss.sum()
        elif reduction == 'none': #(b,h,w)
            return focal_loss 
            

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=1, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.eps = eps

    def forward(self, y_pred, y_true, weight=None, reduction='mean'): 
        y_true = F.one_hot(y_true, NUM_CLASSES).permute(0,3,1,2) #(b,h,w)->(b,c,h,w)
        y_pred = F.softmax(y_pred, dim=1)  #logits转为概率
        inter = torch.sum(y_pred*y_true, dim=[2,3]) #(b,c) 每个类别预测正确的个数
        combined = torch.sum(y_pred.pow(self.p)+y_true.pow(self.p), dim=[2,3])
        dice_loss = 1 - (2 * inter + self.smooth)/(combined + self.smooth+ self.eps) #(b,c)
        
        if weight is not None:
            dice_loss = dice_loss*weight 
        if reduction == 'mean':
            return dice_loss.mean() #求类别均值，大概1左右
            # return dice_loss.sum(dim=1).mean() #求样本均值，相当于类别均值乘以类别数8，和cross_entropy得到的loss更接近
        elif reduction == 'sum':
            return dice_loss.sum()
        elif reduction == 'none': #(b,c)
            return dice_loss


def get_criterion(name):
    if name=='cross_entropy': #以pixel为单位，dice以样本为单位
        criterion = F.cross_entropy
    elif name=='focal_loss':
        criterion = FocalLoss()
    elif name=='dice_loss':
        criterion = DiceLoss()    
    return criterion


if __name__=='__main__':
    y_pred = torch.rand(2,8,5,5)
    y_true = torch.randint(0,8,[2,5,5])
    criterion = get_criterion('cross_entropy')
    print(criterion(y_pred, y_true, WEIGHT))