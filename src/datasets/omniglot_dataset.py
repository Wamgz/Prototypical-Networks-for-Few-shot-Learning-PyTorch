# coding=utf-8
import torch.utils.data as data
from PIL import Image
import numpy as np
import shutil
import errno
import torch
import os
import logging
from logger_utils import Logger

'''
Inspired by https://github.com/pytorch/vision/pull/46
'''

IMG_CACHE = {}

class OmniglotDataset(data.Dataset):
    vinalys_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinalys_baseurl + 'test.txt',
        'train': vinalys_baseurl + 'train.txt',
        'trainval': vinalys_baseurl + 'trainval.txt',
        'val': vinalys_baseurl + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='../data/omniglotDataset', transform=None, target_transform=None, download=True):
        '''
        The items are (filename,category). The index of all the categories can be found in self.idx_classes
        Args:
        - root: the directory where the data will be stored
        - transform: how to transform the input
        - target_transform: how to transform the target
        - download: need to download the data
        '''
        super(OmniglotDataset, self).__init__()
        self.root = root # ../data
        self.transform = transform # None
        self.target_transform = target_transform # None

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')
        self.classes = get_current_classes(os.path.join(
            self.root, self.splits_folder, mode + '.txt'))
        self.all_items = find_items(os.path.join(
            self.root, self.processed_folder), self.classes) # size: 13760, list: [('0739_02.png', 'Mkhedruli_(Georgian)/character11', '../data/data/Mkhedruli_(Georgian)/character11', '/rot000'), ····]

        self.idx_classes = index_classes(self.all_items) # {key: 'Mkhedruli_(Georgian)/character11/rot000', value: 0}

        paths, self.y = zip(*[self.get_path_label(pl)
                              for pl in range(len(self))]) #paths: list, size: 13760, 格式：'../data/data/Mkhedruli_(Georgian)/character11/0739_02.png/rot000'，表示图片路径
        #labels: list, size: 13760, 格式: 1(int)，表示上面的path的图片对应的标签

        self.x = map(load_img, paths, range(len(paths)))
        self.x = list(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.all_items)

    def get_path_label(self, index):
        filename = self.all_items[index][0] #'0739_02.png'
        rot = self.all_items[index][-1] #'/rot000'
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index]
                                  [1] + self.all_items[index][-1]]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            logger.info('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        for url in self.urls:
            logger.info('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            orig_root = os.path.join(self.root, self.raw_folder)
            logger.info("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        logger.info("Download finished.")


def find_items(root_dir, classes):
    retour = [] #('0459_14.png', 'Gujarati/character42', '../data/data/Gujarati/character42', '/rot000')
    rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            r = root.split(os.sep)
            lr = len(r)
            label = r[lr - 2] + os.sep + r[lr - 1] #'Gujarati/character42'
            for rot in rots:
                if label + rot in classes and (f.endswith("png")):
                    retour.extend([(f, label, root, rot)])
    logger.info("== Dataset: Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items: #i: ('0459_14.png', 'Gujarati/character42', '../data/data/Gujarati/character42', '/rot000')
        if (not i[1] + i[-1] in idx):
            idx[i[1] + i[-1]] = len(idx)
    logger.info("== Dataset: Found %d classes" % len(idx))
    return idx


def get_current_classes(fname):
    with open(fname) as f:
        classes = f.read().replace('/', os.sep).splitlines()
    return classes

# 加载path指向的路径的图片，并且对图片进行旋转
def load_img(path, idx):
    path, rot = path.split(os.sep + 'rot')
    if path in IMG_CACHE:
        x = IMG_CACHE[path]
    else:
        x = Image.open(path)
        IMG_CACHE[path] = x
    x = x.rotate(float(rot))
    x = x.resize((28, 28))

    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x) #TODO 为什么要这么做
    x = x.transpose(0, 1).contiguous().view(shape)

    return x
