import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from ImageDepthNet import ImageDepthNet
from torch.utils import data
import numpy as np
import os

trainset = 'ISTD/ISTD'
data_root = './Data/'
img_size = 224
train_dataset = get_loader(trainset, data_root, img_size, mode='train')

train_loader = data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=1)

for i, data_batch in enumerate(train_loader):
    images, label_224, label_14, label_28, label_56, label_112 = data_batch
    print(label_224)
    break