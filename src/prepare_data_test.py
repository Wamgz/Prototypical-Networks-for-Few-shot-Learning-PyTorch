from src.datasets.miniimagenet import MiniImageNet
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from src.datasets.omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser_util import get_parser
from src.datasets.miniimagenet import MiniImageNet
from tqdm import tqdm
import numpy as np
import torch
import os

if __name__ == '__main__':
    options = get_parser().parse_args()

    dataset = MiniImageNet('train', options)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    for i, data in enumerate(dataloader):
        x, y = data
        print(x.shape, y.shape)