# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import adabound as abd

from torchcontrib.optim.swa import SWA

from tqdm import tqdm

import utils
from datasets import load_dataset
from models import ShakeResNet, ShakeResNeXt




def main(args):
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    train_loader, test_loader = load_dataset(args.label, args.batch_size, args.half_length, args.nholes)
    
    if args.label == 10:
        model = ShakeResNet(args.depth, args.w_base, args.label)
    else:
        model = ShakeResNeXt(args.depth, args.w_base, args.cardinary, args.label)
    
    model = torch.nn.DataParallel(model).cuda()
    
    cudnn.benckmark = True
    
    def schedule(epoch):
        t = (epoch) / (args.swa_start if args.optimizer=='swa' else args.epochs)
        lr_ratio = args.swa_lr / args.lr_init if args.optimizer=='swa' else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return args.lr_init * factor
    
    if args.optimizer=='sdg':
        opt = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.weight_decay,nesterov=args.nesterov)
    
    if args.optimizer == 'abd':
        opt= abd.AdaBound(model.parameters(), lr=args.lr, gamma= 0.9, weight_decay=args.weight_decay, final_lr=0.1)
    
    loss_func = nn.CrossEntropyLoss().cuda()

    if args.optimizer=='swa':
        optimizer=optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, wright_decay=args.weight_decay)
        steps_per_epoch = len(train_loader.dataset) / args.batch_size
        steps_per_epoch = int(steps_per_epoch)
        optimizer = SWA(optimizer, swa_start=args.swa_start * steps_per_epoch,swa_freq=steps_per_epoch, swa_lr=args.swa_lr)

    

    headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."]
    
    if args.optimizer=='swa':
        headers = headers[:-1] + ['swa_te_loss', 'swa_te_acc'] + headers[-1:]
        swa_res = {'loss': None, 'accuracy': None}
    
    logger = utils.Logger(args.checkpoint, headers)

    for e in range(args.epochs):
        if args.optimizer=='swa':
            lr= schedule(epoch)
            utils.adjust_learning_rate(optimizer, lr)
        else:    
            lr = utils.cosine_lr(opt, args.lr, e, args.epochs)
        
        #train
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(train_loader), leave=False)
        
        for x, t in train_loader:
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += utils.accuracy(y, t).item()
            train_loss += loss.item() * t.size(0)
            train_n += t.size(0)
            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                train_loss / train_n, train_acc / train_n * 100), refresh=True)
            bar.update()
        bar.close()
        
        #eval
        
        model.eval()
        test_loss, test_acc, test_n = 0, 0, 0
        
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            with torch.no_grad():
                x, t = Variable(x.cuda()), Variable(t.cuda())
                y = model(x)
                loss = loss_func(y, t)
                test_loss += loss.item() * t.size(0)
                test_acc += utils.accuracy(y, t).item()
                test_n += t.size(0)
        logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                     train_acc / train_n * 100, test_acc / test_n * 100)
        
        if args.optimizer =='swa'  and (epoch + 1) >= args.swa_start and args.eval_freq>1:
            if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
                optimizer.swap_swa_sgd()
                optimizer.bn_update(train_loaders, model, device='cuda')
                optimizer.swap_swa_sgd()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint")
    # For Networks
    parser.add_argument("--depth", type=int, default=29)
    parser.add_argument("--w_base", type=int, default=64)
    parser.add_argument("--cardinary", type=int, default=4)
    # For Training
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--batch_size", type=int, default=32)
    #cutout
    parser.add_argument("--half_length", type=int, default=8)
    parser.add_argument("--nholes", type=int, default=1)
    #new-optimizers
    parser.add_argument("--optimizer", type=str, default='sdg')
    parser.add_argument('--swa_start', type=float, default=161)
    parser.add_argument('--swa_lr', type=float, default=0.05)
    #BatchEVAL
    parser.add_argument('--eval_freq', type= int, default=1)
    
    args = parser.parse_args()
    main(args)
