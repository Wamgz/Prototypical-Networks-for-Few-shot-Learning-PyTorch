# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from datasets.omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from parser_util import get_parser
from datasets.miniimagenet import MiniImageNet
from datasets.stanfordCars import StanfordCars
from tqdm import tqdm
import numpy as np
import torch
import os
import sys
from logger_utils import logger
from Vit import ViT

if os.curdir not in sys.path:
    sys.path.append(os.curdir)

options = get_parser().parse_args()
device = torch.device(options.cuda)


def init_dataset(opt, mode):
    if opt.dataset_name == 'omniglotDataset':
        dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
        _dataset_exception_handle(dataset=dataset, n_classes=len(np.unique(dataset.y)), mode=mode, opt=opt)
        return dataset
    elif opt.dataset_name == 'miniImagenet':
        dataset = MiniImageNet(mode=mode, opt=options)
        _dataset_exception_handle(dataset=dataset, n_classes=len(np.unique(dataset.y)), mode=mode, opt=opt)
        return dataset
    elif opt.dataset_name == 'stanfordCars':
        dataset = StanfordCars(mode=mode, opt=options)
        _dataset_exception_handle(dataset=dataset, n_classes=len(np.unique(dataset.y)), mode=mode, opt=opt)
        return dataset

    raise ValueError('Unsupported dataset_name {}'.format(opt.dataset_name))


# region ##数据加载失败抛异常处理##
def _dataset_exception_handle(dataset, n_classes, mode, opt):
    n_classes = len(np.unique(dataset.y))
    if mode == 'train' and n_classes < opt.classes_per_it_tr or mode == 'val' and n_classes < opt.classes_per_it_val:
        raise (Exception('There are not enough classes in the data in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))


# endregion

def init_sampler(opt, labels, mode, dataset_name='miniImagenet'):
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


def init_model(opt):
    '''
    Initialize the ProtoNet
    '''
    if opt.model_name == 'cnn':
        return ProtoNet(x_dim=opt.channel).to(device)
    elif opt.model_name == 'vit':
        return ViT(
            image_size=128,
            patch_size=32,
            out_dim=1600,
            dim=256,
            depth=2,
            heads=8,
            dim_head=64,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            channels=3
        ).to(device)

    raise ValueError('Unsupported model_name {}'.format(opt.model_name))


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0


    best_model_path = os.path.join(opt.experiment_root, opt.dataset_name + '_' + opt.model_name + '_' + 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, opt.dataset_name + '_' + opt.model_name + '_' + 'last_model.pth')

    for epoch in range(opt.epochs):
        logger.info('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch  # x: (batch, C, H, W), y:(batch, )
            x, y = x.to(device), y.to(device)
            model_output = model(x)  # (batch, z_dim(protonet的超参数决定))
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr,
                                n_query=opt.num_query_tr)
            loss.backward()  # tensor(254.0303, grad_fn=<NegBackward0>)
            optim.step()
            train_loss.append(loss.detach())
            train_acc.append(acc.detach())
        avg_loss = torch.tensor(train_loss[-opt.iterations:]).mean()
        avg_acc = torch.tensor(train_acc[-opt.iterations:]).mean()
        logger.info('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)
                model_output = model(x)
                loss, acc = loss_fn(model_output, target=y,
                                    n_support=opt.num_support_val,
                                    n_query=opt.num_query_val)
                val_loss.append(loss.detach())
                val_acc.append(acc.detach())
            avg_loss = torch.tensor(val_loss[-opt.iterations:]).mean()
            avg_acc = torch.tensor(val_acc[-opt.iterations:]).mean()
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
            logger.info('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val,
                             n_query=opt.num_query_val)
            avg_acc.append(acc.detach())
    avg_acc = torch.tensor(avg_acc).mean()
    logger.info('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''

    test_dataloader = init_dataset(options)[-1]
    model = init_model(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    '''
    Initialize everything and train
    '''
    '''
    Namespace(dataset_root='../data', experiment_root='../output', epochs=100, learning_rate=0.001, lr_scheduler_step=20, lr_scheduler_gamma=0.5, iterations=100, classes_per_it_tr=60, num_support_tr=5, num_query_tr=5, classes_per_it_val=5, num_support_val=5, num_query_val=15, manual_seed=7, cuda=False)
    '''
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    test_dataloader = init_dataloader(options, 'test')

    model = init_model(options)

    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    logger.info('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    logger.info('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)


if __name__ == '__main__':
    main()
