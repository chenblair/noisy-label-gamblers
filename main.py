# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from data.mnist import MNIST
from data.cifar import CIFAR10
from model import CNN_basic, CNN_small, LSTMClassifier
from optimizer import LaProp
import argparse
import sys
import numpy as np
import datetime
import shutil
import load_data
import json
import pdb

def train(args,
          model,
          device,
          train_loader,
          optimizer,
          epoch,
          num_classes=10,
          use_gamblers=True,
          text=False):
    model.train()
    loss_a = []

    for batch_idx, batch in enumerate(train_loader):
        if text:
            data = batch.text[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if (data.size()[0] != args.batch_size):
                continue
        else:
            data, target = batch[0], batch[1]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = F.softmax(output, dim=1)

        with torch.no_grad():
            # default to nll
            # if eps > num_classes, model learning is equivalent to nll
            eps = 1000 + num_classes

            # set lambda in the gambler's loss
            if (args.lambda_type == 'euc'):
                eps = ((1 - output[:, num_classes])**2 + 1e-10) / (torch.sum(
                    (output[:, :num_classes])**2, (1, -1)))
            elif (args.lambda_type == 'mid'):
                eps = ((1 - output[:, num_classes]) + 1e-10) / (torch.sum(
                    (output[:, :num_classes])**2, (1, -1)))
            elif (args.lambda_type == 'exp'):
                columns = output[:, :num_classes] + 1E-10
                eps = torch.exp(
                    -1 * (torch.sum(torch.log(columns) * columns,
                                    (1, -1)) + 1E-10) / torch.sum(
                                        columns, (1, -1)))
            elif (args.lambda_type == 'gmblers'):
                eps = args.eps

        if (args.lambda_type == 'lq'):
            q = 0.7
            loss = -1 * F.nll_loss(output, target, reduction='none')
            loss = (1 - (loss + 1e-10).pow(q)) / q
            loss = torch.mean(loss)
        # compute gambler's loss
        elif (use_gamblers and args.lambda_type != 'nll'):
            output = (output + (output[:, num_classes] / eps).unsqueeze(1) +
                  1E-10).log()
            loss = F.nll_loss(output, target)
        else:
            output = output.log()
            loss = F.nll_loss(output, target)
        loss_a.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(loss_a)


def test(args, model, device, test_loader, num_classes, text=False):
    model.eval()
    test_loss = 0
    correct = 0
    acc = []
    with torch.no_grad():
        for batch in test_loader:
            if text:
                data = batch.text[0]
                target = batch.label
                target = torch.autograd.Variable(target).long()
                if (data.size()[0] != args.batch_size):
                    continue
            else:
                data, target = batch[0], batch[1]

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()  # sum up batch loss
            pred = output[:, :num_classes].argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), acc))
    return acc, test_loss


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Gambler\'s Loss Runner')

    parser.add_argument(
        '--result_dir',
        type=str,
        help='directory to save result txt files',
        default='results')
    parser.add_argument(
        '--noise_rate',
        type=float,
        help='corruption rate, should be less than 1',
        default=0.0)
    parser.add_argument(
        '--noise_type',
        type=str,
        help='[pairflip, symmetric]',
        default='symmetric')
    parser.add_argument(
        '--dataset', type=str, help='mnist, cifar10, or imdb', default='mnist')
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='how many subprocesses to use for data loading')
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument(
        '--load_model',
        action='store_true',
        default=False)
    parser.add_argument(
        '--use_scheduler',
        action='store_true',
        default=False)
    parser.set_defaults(load_model=False)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        metavar='N',
        help='input batch size for training (default: 128)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        metavar='N',
        help='how many batches to wait before logging training status (default: 100)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate (default: 0.001)')
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        default=False,
        help='enables early stopping criterion for only symmetric datasets')
    parser.add_argument('--eps', type=float, help='set lambda for lambda type \'gmblers\' only', default=1000.0)
    parser.add_argument(
        '--lambda_type', type=str, help='[nll, euc, mid, exp, gmblers]', default="euc")
    parser.add_argument(
        '--start_gamblers', type=int, help='number of epochs before starting gamblers', default=0)
    parser.add_argument(
        '--gpu', type=int, help='gpu index', default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'mnist':
        input_channel = 1
        num_classes = 10
        train_dataset = MNIST(
            root='./data/',
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        test_dataset = MNIST(
            root='./data/',
            download=True,
            train=False,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=False)

    if args.dataset == 'cifar10':
        input_channel = 3
        num_classes = 10
        train_dataset = CIFAR10(
            root='./data/',
            download=True,
            train=True,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        test_dataset = CIFAR10(
            root='./data/',
            download=True,
            train=False,
            transform=transforms.ToTensor(),
            noise_type=args.noise_type,
            noise_rate=args.noise_rate)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=False)

    if args.dataset == 'imdb':
        num_classes = 2
        embedding_length = 300
        hidden_size = 256
        print('loading dataset...')
        TEXT, vocab_size, word_embeddings, train_loader, valid_iter, test_loader = load_data.load_dataset(
            rate=args.noise_rate, batch_size=args.batch_size)

    if args.dataset == 'animal-10n':
        data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_channel = 3
        num_classes = 10
        train_dataset = datasets.ImageFolder('data/animal-10n/train', transform=data_transform)
        test_dataset = datasets.ImageFolder('data/animal-10n/test', transform=data_transform)
        print('loading dataset...')
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            shuffle=False)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_cuda else "cpu")
    print("using {}".format(device))

    print('building model...')
    if args.dataset == 'mnist':
        model = CNN_basic(num_classes=num_classes + 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.dataset == 'cifar10':
        model = CNN_small(num_classes=num_classes + 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.dataset == 'imdb':
        model = LSTMClassifier(args.batch_size, num_classes + 1, hidden_size,
                               vocab_size, embedding_length, word_embeddings).to(device)
        optimizer = LaProp(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.dataset == 'animal-10n':
        model = models.vgg19_bn(num_classes=num_classes + 1, dropout=0.2).to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0.0005, nesterov=True)
        if (args.use_scheduler):
            change_lr = lambda epoch: pow(0.2, int(epoch >= 50) + int(epoch >= 75))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=change_lr)

    test_accs = []
    train_losses = []
    test_losses = []
    out = []

    if args.early_stopping:
        eee = 1 - args.noise_rate
        criteria = (-1) * (eee * np.log(eee) + (1 - eee) * np.log(
            (1 - eee) / (args.eps - 1)))

    name = "{}_{}_{}_{:.2f}_{}_{}".format(args.dataset, args.lambda_type, args.noise_type, args.noise_rate, args.eps, args.seed)

    if args.load_model:
        checkpoint = torch.load("{}/{}.npy".format(args.result_dir, name))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint["train_loss"]
        test_losses = checkpoint["test_loss"]
        test_accs = checkpoint["test_acc"]
        print("loaded {}".format(name))
        print("{} epochs so far".format(len(train_losses)))

    if not os.path.exists(args.result_dir):
        os.system('mkdir -p %s' % args.result_dir)

    for epoch in range(1, args.n_epoch + 1):
        print(name)
        for param_group in optimizer.param_groups:
            print(epoch, param_group['lr'])
        train_loss = train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            num_classes=num_classes,
            use_gamblers=(epoch > args.start_gamblers),
            text=(args.dataset == 'imdb'))
        train_losses.append(train_loss)
        test_acc, test_loss = test(args, model, device, test_loader, num_classes, text=(args.dataset == 'imdb'))
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        
        if (str(train_loss) == 'nan'):
            print('nan, breaking')
            break
        
        if args.early_stopping and epoch >= args.start_gamblers and train_loss < criteria :
            print('epoch: {}, loss fulfilled early stopping criteria: {} < {}'.format(epoch, train_loss, criteria))
            break
        
        model_data = {
            'command': " ".join(sys.argv),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "train_loss" : train_losses,
            "test_loss" : test_losses,
            "test_acc" : test_accs
        }
        if (args.use_scheduler):
            scheduler.step()
            model_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(model_data, "{}/{}.npy".format(args.result_dir, name))

    save_data = {
        "train_loss" : train_losses,
        "test_loss" : test_losses,
        "test_acc" : test_accs
    }
    save_file = "{}/{}.json".format(args.result_dir, name)
    json.dump(save_data, open(save_file, 'w'))

if __name__ == '__main__':
    main()
