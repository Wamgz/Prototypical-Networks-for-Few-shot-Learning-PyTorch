from __future__ import print_function
import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
import csv
import torch.utils.data as data
from torchvision import transforms as transforms

class MiniImageNet(data.Dataset):
    '''
    类别数量：100 个类别，每个类别 600 张图片，共计 60,000 张图片。
    数据内容：RGB 图片，.jpg 格式，分辨率 84x84。
    数据切分：训练集 64 个类，验证集 16 个类，测试集 20 个类。
    '''
    def __init__(self, mode, opt, transform=None, target_transform=None):
        self.im_width, self.im_height, self.channels = 3, 84, 84
        self.split = mode
        self.root = opt.dataset_root
        self.x = []
        self.y = []
        if transform is None: # 使用默认的transform
            tsfm = []
            tsfm.append(transforms.ToPILImage())

            if opt.height > 0 and opt.width > 0:
                self.im_width, self.im_height, self.channels = opt.width, opt.height, opt.channel
                tsfm.append(transforms.Resize((self.im_height, self.im_width)))

            tsfm.append(transforms.ToTensor())
            tsfm.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)))
            self.transform = transforms.Compose(tsfm)

        #1、读取pkl文件
        pkl_name = '{}/miniImagenet/data/mini-imagenet-cache-{}.pkl'.format(self.root, self.split)
        print('Loading pkl data: {} '.format(pkl_name))

        try:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f, encoding='bytes')
                image_data = data[b'image_data']
                class_dict = data[b'class_dict']
        except:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f)
                image_data = data['image_data']  # (38400, 84, 84, 3)
                class_dict = data['class_dict']  # dict, key_num = 64, {'n01532829': [0, 1, 2···599]}
        self.x = image_data
        self.y = np.arange(len(class_dict.keys())).reshape(1, -1).repeat(600, 1).squeeze(0)


    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

    def load_data_pkl(self):
        """
            load the pkl processed mini-imagenet into label,unlabel
        """

        pkl_name = '{}/miniImagenet/data/mini-imagenet-cache-{}.pkl'.format(self.root, self.split)
        print('Loading pkl data: {} '.format(pkl_name))

        try:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f, encoding='bytes')
                image_data = data[b'image_data']
                class_dict = data[b'class_dict']
        except:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f)
                image_data = data['image_data']  # (38400, 84, 84, 3)
                class_dict = data['class_dict']  # dict, key_num = 64, {'n01532829': [0, 1, 2···599]}
        data_classes = sorted(class_dict.keys())
        for i, cls in enumerate(data_classes):
            idxs = class_dict[cls]
            np.random.RandomState(self.seed).shuffle(idxs)  # fix the seed to keep label,unlabel fixed
            self.x[i] = image_data[idxs]

        print(data.keys(), image_data.shape, class_dict.keys())
        data_classes = sorted(class_dict.keys())  # sorted to keep the order

        n_classes = len(data_classes) # 64
        print('n_classes:{}, n_label:{}'.format(n_classes, self.n_label))
        dataset_l = np.zeros([n_classes, self.n_label, self.im_height, self.im_width, self.channels], dtype=np.float32) #(64, 600, 84, 84, 3) n_label是每个class下的sample个数
        if self.n_unlabel > 0:
            dataset_u = np.zeros([n_classes, self.n_unlabel, self.im_height, self.im_width, self.channels],
                                 dtype=np.float32)
        else:
            dataset_u = []
        # 每个class下600个sample
        for i, cls in enumerate(data_classes):
            idxs = class_dict[cls]
            np.random.RandomState(self.seed).shuffle(idxs)  # fix the seed to keep label,unlabel fixed
            dataset_l[i] = image_data[idxs[0:self.n_label]]
            if self.n_unlabel > 0:
                dataset_u[i] = image_data[idxs[self.n_label:]]
        print('labeled data:', np.shape(dataset_l))
        print('unlabeled data:', np.shape(dataset_u))

        self.x = dataset_l
        self.dataset_u = dataset_u
        self.n_classes = n_classes

        del image_data

    def next_data(self, n_way, n_shot, n_query, num_unlabel=0, n_distractor=0, train=True):
        """
            get support,query,unlabel data from n_way
            get unlabel data from n_distractor
        """
        support = np.zeros([n_way, n_shot, self.im_height, self.im_width, self.channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, self.im_height, self.im_width, self.channels], dtype=np.float32)
        if num_unlabel > 0:
            unlabel = np.zeros([n_way + n_distractor, num_unlabel, self.im_height, self.im_width, self.channels],
                               dtype=np.float32)
        else:
            unlabel = []
            n_distractor = 0
        # 取哪几个class作为support和query set
        selected_classes = np.random.permutation(self.n_classes)[:n_way + n_distractor]
        for i, cls in enumerate(selected_classes[0:n_way]):  # train way
            # labled data
            idx1 = np.random.permutation(self.n_label)[:n_shot + n_query] # 从trainset里取出support set和query set
            support[i] = self.x[cls, idx1[:n_shot]]
            query[i] = self.x[cls, idx1[n_shot:]]
            # unlabel
            if num_unlabel > 0:
                idx2 = np.random.permutation(self.n_unlabel)[:num_unlabel]
                unlabel[i] = self.dataset_u[cls, idx2]

        for j, cls in enumerate(selected_classes[self.n_classes:]):  # distractor way
            idx3 = np.random.permutation(self.n_unlabel)[:num_unlabel]
            unlabel[i + j] = self.dataset_u[cls, idx3]

        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        # unlabel_labels = np.tile(np.arange(n_way+n_distractor)[:, np.newaxis], (1, num_unlabel)).astype(np.uint8)

        return support, support_labels, query, query_labels, unlabel
