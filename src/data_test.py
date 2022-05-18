from src.datasets.omniglot_dataset import OmniglotDataset
import torch
import numpy as np
from prototypical_batch_sampler import PrototypicalBatchSampler
from parser_util import get_parser
from tqdm import tqdm
from Vit import ViT


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_dataset(opt, mode):
    dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.y))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the data in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader

if __name__ == '__main__':
    # v = ViT(
    #     image_size=256,
    #     patch_size=32,
    #     num_classes=1000,
    #     dim=1024,
    #     depth=6,
    #     heads=16,
    #     mlp_dim=2048,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    #
    # img = torch.randn(1, 3, 256, 256)
    #
    # preds = v(img)  # (1, 1000)
    opt = get_parser().parse_args()
    tr_dataloader = init_dataloader(opt, 'train')
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=160,
        dim=1024,
        depth=1,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    for epoch in range(opt.epochs):
        logger.info('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        for batch in tqdm(tr_iter):
            x, y = batch # x: (600, 1, 256, 256), y:(600, )
            x, y = x[:5, :, :, :], y[:5]
            out = model(x)
            logger.info(out.shape)
