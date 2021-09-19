from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
import os
from model import Model
from dataset import Dataset
from train import train
from test import test
import option


def setup_seed(seed):
    torch.manual_seed(seed)  # 为CPU中设置种子，生成随机数
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子，生成随机数
    np.random.seed(seed)  # 用于生成指定随机数
    random.seed(seed)
    # 当seed()没有参数时，每次生成的随机数是不一样的，
    # 而当seed()有参数时，每次生成的随机数是一样的，同时选择不同的参数生成的随机数也不一样
    torch.backends.cudnn.deterministic = True
    # 将这个
    # flag
    # 置为True的话，每次返回的卷积算法将是确定的，即默认算法。
    # 如果配合上设置
    # Torch
    # 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # 多进程 设置进程启动方式
    # setup_seed(2333)
    args = option.parser.parse_args()
    # 将参数字符串转换为对象并将其设为命名空间的属性。 返回带有成员的命名空间。
    device = torch.device("cuda")
    # DataLoader和Dataset是pytorch中数据读取的核心
    # torch.utils.data.DataLoader
    # 功能：构建可迭代的数据装载器；
    # dataset: Dataset类，决定数据从哪里读取及如何读取；
    # batchsize：设定每次训练迭代时加载的数据量；
    # num_works: 是否多进程读取数据；
    # shuffle：每个epoch是否乱序，是否将数据打乱；
    # drop_last：当样本数不能被batchsize整除时，是否舍弃最后一批数据；

    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    device = torch.device('cuda:{}'.format(args.gpus) if args.gpus != '-1' else 'cpu')
    # 模型加载到指定设备上
    model = Model(args).to(device)

    for name, value in model.named_parameters():
        print(name)
        # map()是Python内置的高阶函数，它接收一个函数f和一个list，
        # 并通过把函数f依次作用在list的每个元素上，
        # 得到一个新的list，并返回。


    # list[0][2]
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    #   base_param将使用args.lr的学习率
    #   model.approximator.parameters()将使用args.lr / 2的学习率
    #   model.conv1d_approximator.parameters()将使用args.lr / 2的学习率
    optimizer = optim.Adam([{'params': base_param}, {'params': model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2}, ], lr=args.lr, weight_decay=0.000)

    # 学习率调整方法 本代码使用有序调整 MultiStepLR
    # 调节的epoch是自己定义，无须一定是【30， 60， 90】 这种等差数列；
    # 请注意，这种衰减是由外部的设置来更改的。 当last_epoch=-1时，将初始LR设置为LR。
    # 下面代码的意思是 在epoch=10时改变学习率，变为之前的0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    criterion = torch.nn.BCELoss()

    is_topk = True
    gt = np.load(args.gt)
    pr_auc, pr_auc_online = test(test_loader, model, device, gt)
    print('Random initalization: offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    for epoch in range(args.max_epoch):
        scheduler.step()
        st = time.time()
        train(train_loader, model, optimizer, criterion, device, is_topk)
        if epoch % 2 == 0 and not epoch == 0:
            torch.save(model.state_dict(), './ckpt/'+args.model_name+'{}.pkl'.format(epoch))

        pr_auc, pr_auc_online = test(test_loader, model, device, gt)
        print('Epoch {0}/{1}: offline pr_auc:{2:.4}; online pr_auc:{3:.4}\n'.format(epoch, args.max_epoch, pr_auc, pr_auc_online))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
