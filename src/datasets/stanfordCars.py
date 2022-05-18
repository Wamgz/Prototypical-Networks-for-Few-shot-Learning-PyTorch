from __future__ import print_function
import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
import csv
import torch.utils.data as data
from torchvision import transforms as transforms
import scipy.io
from src.parser_util import get_parser


class StanfordCars(data.Dataset):
    '''
    一共包含16185张不同型号的汽车图片，其中8144张为训练集，8041张为测试集
    196个类，train和test每个class的sample数量不低于60
    '''
    def __init__(self, mode, opt, transform=None, target_transform=None):
        self.im_width, self.im_height, self.channels = opt.width, opt.height, opt.channel
        self.split = mode
        self.root = opt.dataset_root
        self.x = []
        self.y = []
        if transform is None: # 使用默认的transform
            tsfm = []
            if opt.height > 0 and opt.width > 0:
                self.im_width, self.im_height, self.channels = opt.width, opt.height, opt.channel
                tsfm.append(transforms.Resize((self.im_height, self.im_width)))

            tsfm.append(transforms.ToTensor())
            tsfm.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)))
            self.transform = transforms.Compose(tsfm)
        dir = os.path.join(self.root, opt.dataset_name)
        data_dir = os.path.join(dir, 'data')
        mode2image = {'train': [], 'val': [], 'test': []} ## TODO 如何划分
        image2label = {}
        with open(os.path.join(dir, 'mat2txt.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                split = line.strip().split(' ')
                cur_mode = 'train' if split[2] == '0' else 'test'
                img_path = os.path.join(os.path.join(data_dir, split[0]))
                mode2image[cur_mode].append(img_path)
                image2label[img_path] = int(split[1]) - 1
            mode2image['val'] = mode2image['test']
        self.x = mode2image[mode]
        self.y = [image2label[path] for path in self.x]
    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.open(x)
        if len(x.split()) < 3:
            x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    options = get_parser().parse_args()
    dataset = StanfordCars('train', options)

