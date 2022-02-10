from torch import optim
from config import *

def get_optimizer(model, name='sgd'):
    if name=='sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    elif name=='adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr =LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return optimizer


def adjust_learning_rate(optimizer, epoch):
    if epoch < 3:
        lr = LEARNING_RATE
    elif epoch < 6:
        lr = 3e-4
    elif epoch < 8:
        lr = 5e-5
    else:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
