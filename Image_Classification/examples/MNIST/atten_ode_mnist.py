import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pdb
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

## written by sheoyon
# written by original
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str,
                    choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False,
                    choices=[True, False])  # True : ADJOINT
parser.add_argument('--downsampling-method', type=str,
                    default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval,
                    default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
# for attentive layer regularization
parser.add_argument('--regul', type=str,
                    choices=['none', 'l1', 'l2'], default='none')
parser.add_argument('--lam', type=float, default=1e-2)

parser.add_argument('--save', type=str,
                    default='./mnist_result/experiment')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type = int, default = 2021)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    print("adjoint")
else:
    from torchdiffeq import odeint
    print("odeint")


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class Downsampling(nn.Module):
    def __init__(self, dim):
        super(Downsampling, self).__init__()
        self.conv1 = nn.Conv2d(1, dim, 3, 1)
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 4, 2, 1)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim, dim, 4, 2, 1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out


class atten_init(nn.Module):

    def __init__(self, dim):
        super(atten_init, self).__init__()
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.norm3 = norm(dim)

    def forward(self, x):

        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.norm3(out)

        return out


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)

        self.norm1_at = norm(dim)
        self.relu_at = nn.ReLU(inplace=True)
        self.conv1_at = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2_at = norm(dim)
        self.conv2_at = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3_at = norm(dim)

        self.sigmoid1 = nn.Sigmoid()
        self.nfe = 0

    def forward(self, t, x):
        size = x.shape
        size = int(size[0]/2)
        h_0 = x[:size]
        a_0 = x[size:]
        a_s = self.sigmoid1(a_0)
        h_ = torch.mul(h_0, a_s)

        self.nfe += 1

        out1 = self.norm1(h_)
        out1 = self.relu(out1)
        out1 = self.conv1(t, out1)
        out1 = self.norm2(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(t, out1)
        out1 = self.norm3(out1)

        out2 = self.norm1_at(h_)
        out2 = self.relu_at(out2)
        out2 = self.conv1_at(t, out2)
        out2 = self.norm2_at(out2)
        out2 = self.relu_at(out2)
        out2 = self.conv2_at(t, out2)
        out2 = self.norm3_at(out2)
        out = torch.cat([out1, out2])
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, atten_init):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.atten_init = atten_init

        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):

        self.integration_time = self.integration_time.type_as(x)
        a_0 = self.atten_init(x)

        x_0 = torch.cat([x, a_0])
        out = odeint(self.odefunc, x_0, self.integration_time,
                     rtol=args.tol, atol=args.tol)

        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True,
                       download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False,
                       download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):

    total_correct = 0
    for x, y in dataset_loader:

        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        size = x.shape
        size = int(size[0])
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(
            model(x)[:size].cpu().detach().numpy(), axis=1)

        total_correct += np.sum(predicted_class == target_class)
        
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    randomSeed = args.seed
    torch.manual_seed(randomSeed)
    torch.cuda.manual_seed(randomSeed)
    torch.cuda.manual_seed_all(randomSeed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(randomSeed)
    random.seed(randomSeed)

    makedirs(args.save)

    logger = get_logger(logpath=os.path.join(
        args.save, 'logs'), filepath=os.path.abspath(__file__))

    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'
    is_method = args.downsampling_method == 'conv'
    ## hidden vector h(0) mnist 28 by 28
    if args.downsampling_method == 'conv':
        downsampling_layers = [Downsampling(64)]  # h(0)
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2))
        ]

    feature_layers = [ODEBlock(ODEfunc(64), atten_init(64))] if is_odenet else [
        ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(
        (1, 1)), Flatten(), nn.Linear(64, 10)]
    model = nn.Sequential(*downsampling_layers, *
                          feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(
        count_parameters(model)))  # parameters : 208138
    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) # l2-regularization
    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()

        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        # pdb.set_trace()

        logits = model(x)  # x: 128,1,28,28
        size = logits.shape
        size = int(size[0] / 2)

        logits_h = logits[:size]
        logits_a = logits[size:]
        loss_task = criterion(logits_h, y)

        if args.regul == "none":
            loss = loss_task
        else:
            if args.regul == "l1":
                theta_g = torch.norm(logits_a, p=1)
            elif args.regul == "l2":
                theta_g = torch.norm(logits_a, p=2)
            loss = loss_task + theta_g * args.lam

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(
                        args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
