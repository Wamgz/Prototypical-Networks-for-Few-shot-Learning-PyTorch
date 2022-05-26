import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision import models,datasets
from torch.utils.tensorboard import SummaryWriter

x = torch.FloatTensor([100])
y = torch.FloatTensor([500])

for epoch in range(30):
    x = x * 1.2
    y = y / 1.1
    loss = np.random.random()
    with SummaryWriter(log_dir='./logs', comment='train') as writer: #可以直接使用python的with语法，自动调用close方法
        writer.add_histogram('his/x', x, epoch)
        writer.add_histogram('his/y', y, epoch)
        writer.add_scalar('data/x', x, epoch)
        writer.add_scalar('data/y', y, epoch)
        writer.add_scalar('data/loss', loss, epoch)
        writer.add_scalars('data/data_group', {'x': x,
                                                'y': y}, epoch)
