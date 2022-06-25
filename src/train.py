# coding=UTF-8
import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
from prototypical_loss import prototypical_loss as loss_fn
from datasets.omniglot_dataset import OmniglotDataset
from src.models.protonet import ProtoNet
from src.utils.parser_util import get_parser
from datasets.miniimagenet import MiniImageNet
from datasets.stanfordCars import StanfordCars
from tqdm import tqdm
import numpy as np
import torch
from src.utils.logger_utils import logger
from src.models.vit import ViT
from src.models.vit_for_small_dataset import ViT_small
from src.models.swin_transformer import SwinTransformer
from data_loaders.data_fetchers import DataFetcher
from src.data_loaders.prototypical_batch_sampler import PrototypicalBatchSampler
from torch.utils.tensorboard import SummaryWriter
from src.utils.visdom_utils import new_pane, append2pane
from models.gch import FeedForward as MLP
import time
from visdom import Visdom
import math
options = get_parser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = options.cuda
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

    dataloader_params = {
        'pin_memory': True,
        'num_workers': 8
    }
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **dataloader_params)
    if torch.cuda.is_available():
        dataloader = DataFetcher(dataloader)
    return dataset, dataloader


def init_model(opt):
    '''
    Initialize the ProtoNet
    '''
    if opt.model_name == 'cnn':
        return ProtoNet(x_dim=opt.channel).cuda()
    elif opt.model_name == 'vit':
        return ViT(
            image_size=96,
            patch_size=4,
            out_dim=64,
            embed_dim=64,
            depth=4,
            heads=8,
            dim_head=8,
            mlp_dim=64,
            tsfm_dropout=0.1,
            emb_dropout=0.1,
            use_avg_pool_out=True,
            channels=3
        ).cuda()
    elif opt.model_name == 'vit_small':
        return ViT_small(
            image_size=128,
            patch_size=32,
            out_dim=1600,
            dim=256,
            depth=4,
            heads=8,
            dim_head=64,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            channels=3
        ).cuda()
    elif opt.model_name == 'swin_transformer':
        return SwinTransformer(img_size=opt.height, window_size=4, drop_rate=0.1, attn_drop_rate=0.1, only_feature=True).cuda()

    raise ValueError('Unsupported model_name {}'.format(opt.model_name))


def init_optim(opt, model, mlp=None):
    '''
    Initialize optimizer
    '''
    if opt.optimizer == 'Adam':
        trainable_params = []
        trainable_params.append({'params': model.trainable_params()})
        if mlp != None:
            trainable_params.append({'params': mlp.parameters()})
        return torch.optim.Adam(trainable_params,
                            lr=opt.learning_rate,
                            weight_decay=opt.weight_decay)
    elif opt.optimizer == 'SGD':
        return torch.optim.SGD(params=model.trainable_params(),
                               lr=opt.learning_rate,
                               momentum=0.9)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / opt.warm_up_epochs if epoch <= opt.warm_up_epochs else 0.5 * (
                math.cos((epoch + 1 - opt.warm_up_epochs) / (opt.epochs + 1 - opt.warm_up_epochs) * math.pi) + 1)
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warm_up_with_cosine_lr)
    # return torch.optim.lr_scheduler.StepLR(optimizer=optim,
    #                                        gamma=opt.lr_scheduler_gamma,
    #                                        step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, tr_dataset, val_dataset, mlp=None, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    train_xentropy_loss = []
    val_loss = []
    val_acc = []
    val_xentropy_loss = []

    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root,
                                   opt.dataset_name + '_' + opt.model_name + '_' + 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root,
                                   opt.dataset_name + '_' + opt.model_name + '_' + 'last_model.pth')
    now = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    env = Visdom(env=opt.model_name + '-' + opt.dataset_name)
    train_loss_pane, train_xentropy_pane, train_acc_pane = new_pane(env, 'train_loss' + '_' + str(now)), new_pane(env, 'train_xentropy' + '_' + str(now)), new_pane(env, 'train_acc' + '_' + str(now))
    val_loss_pane, val_xentropy_pane, val_acc_pane = new_pane(env, 'val-loss'  + '_' + str(now)), new_pane(env, 'val_xentropy' + '_' + str(now)), new_pane(env, 'val_acc' + '_' + str(now))

    for epoch in range(opt.epochs):
        logger.info('=== Epoch: {}, Learning Rate : {} === '.format(epoch, optim.state_dict()['param_groups'][0]['lr']))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch  # x: (batch, C, H, W), y:(batch, )
            x, y = x.cuda(), y.cuda()
            model_output = model(x)  # (batch, (num_patch + 1) * z_dim)
            classes = torch.unique(y)
            support_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[opt.num_support_tr:], classes))).view(-1)
            query_idxs = torch.stack(list(map(lambda c: y.eq(c).nonzero()[opt.num_support_tr:], classes))).view(-1)
            loss, x_entropy, acc = loss_fn(model_output, labels=y,
                                n_support=opt.num_support_tr,
                                n_query=opt.num_query_tr,
                                dist=opt.dist,
                                aux_loss=opt.use_aux_loss,
                                scale=opt.balance_scale,
                                use_join_loss=opt.use_join_loss,
                                mlp=mlp)
            total_loss = loss + x_entropy
            total_loss.backward()  # tensor(254.0303, grad_fn=<NegBackward0>)
            optim.step()
            train_xentropy_loss.append(x_entropy.detach())
            train_loss.append(loss.detach())
            train_acc.append(acc.detach())
        train_avg_loss = torch.tensor(train_loss[-opt.iterations:]).mean()
        train_avg_acc = torch.tensor(train_acc[-opt.iterations:]).mean()
        train_avg_x_entropy = torch.tensor(train_xentropy_loss[-opt.iterations:]).mean()
        logger.info('Avg Train Loss: {}, Avg Train Xentropy: {}, Avg Train Acc: {}'.format(train_avg_loss, train_avg_x_entropy, train_avg_acc))
        append2pane(torch.FloatTensor([epoch]), x_entropy, env, train_xentropy_pane)
        append2pane(torch.FloatTensor([epoch]), train_avg_loss, env, train_loss_pane)
        append2pane(torch.FloatTensor([epoch]), train_avg_acc, env, train_acc_pane)
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        val_avg_loss, val_avg_acc = 0., 0.
        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = x.cuda(), y.cuda()
                model_output = model(x)
                loss, x_entropy, acc = loss_fn(model_output, labels=y,
                                    n_support=opt.num_support_val,
                                    n_query=opt.num_query_val,
                                    dist=opt.dist,
                                    aux_loss=opt.use_aux_loss,
                                    scale=opt.balance_scale)
                total_loss = loss + x_entropy
                val_xentropy_loss.append(x_entropy.detach())
                val_loss.append(total_loss.detach())
                val_acc.append(acc.detach())
            val_avg_loss = torch.tensor(val_loss[-opt.iterations:]).mean()
            val_avg_xentropy = torch.tensor(val_xentropy_loss[-opt.iterations:]).mean()
            val_avg_acc = torch.tensor(val_acc[-opt.iterations:]).mean()
            postfix = ' (Best)' if val_avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
            append2pane(torch.FloatTensor([epoch]), x_entropy, env, val_xentropy_pane)
            append2pane(torch.FloatTensor([epoch]), val_avg_loss, env, val_loss_pane)
            append2pane(torch.FloatTensor([epoch]), val_avg_acc, env, val_acc_pane)

            logger.info('Avg Val Loss: {}, Avg Val Xentropy: {}, Avg Val Acc: {}{}'.format(
                val_avg_loss, val_avg_xentropy, val_avg_acc, postfix))
            if val_avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = val_avg_acc
                best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model, test_dataset=None):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.cuda(), y.cuda()
            model_output = model(x)
            _, x_entropy, acc = loss_fn(model_output, labels=y,
                             n_support=opt.num_support_val,
                             n_query=opt.num_query_val,
                             aux_loss=opt.use_aux_loss)
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

    tr_dataset, tr_dataloader = init_dataloader(options, 'train')
    val_dataset, val_dataloader = init_dataloader(options, 'val')
    test_dataset, test_dataloader = init_dataloader(options, 'test')

    model = init_model(options)
    optim = init_optim(options, model)

    ## TODO 这里暂时写死，只能用于vit使用
    mlp = MLP(dim=64, out_dim=64).cuda()
    if options.model_name == 'vit':
        optim = init_optim(options, model, mlp)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler,
                tr_dataset=tr_dataset,
                val_dataset=val_dataset,
                mlp=mlp)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    logger.info('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         test_dataset=test_dataset)

    model.load_state_dict(best_state)
    logger.info('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         test_dataset=test_dataset)

if __name__ == '__main__':
    main()

