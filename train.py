# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import adabound as abd

from swa import swa

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
    
    if args.optimizer=='sgd':
        print("using sgd")
        opt = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum ,weight_decay=args.weight_decay,nesterov=args.nesterov)
    
    elif args.optimizer == 'abd':
        print("using adabound")
        opt= abd.AdaBound(model.parameters(), lr=args.lr, gamma= args.gamma, weight_decay=args.weight_decay, final_lr=args.final_lr)
    
    elif args.optimizer=='swa':
        print("using swa")
        opt=optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        steps_per_epoch = len(train_loader.dataset) / args.batch_size
        steps_per_epoch = int(steps_per_epoch)
        opt = swa(opt, swa_start=args.swa_start * steps_per_epoch,swa_freq=steps_per_epoch, swa_lr=args.swa_lr)
    else:
        print("not valid optimizer")
        exit

    loss_func = nn.CrossEntropyLoss().cuda()

    headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."]
    
    #if args.optimizer=='swa':
     #   headers = headers[:-1] + ['swa_te_loss', 'swa_te_acc'] + headers[-1:]
      #  swa_res = {'loss': None, 'accuracy': None}
    
    logger = utils.Logger(args.checkpoint, headers, mod = args.optimizer)

    for e in range(args.epochs):

        if args.optimizer=='swa':
            lr= utils.schedule(e, args.optimizer, args.epochs, args.swa_start, args.swa_lr, args.lr)
            utils.adjust_learning_rate(opt, lr)
        elif args.optimizer=='sgd':    
            lr = utils.cosine_lr(opt, args.lr, e, args.epochs)
        else:
            exit
        
        #train
        train_loss, train_acc, train_n= utils.train_epoch(train_loader, model, opt)
        #eval
        test_loss, test_acc, test_n = utils.eval_epoch(test_loader, model)
        
        logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                     train_acc / train_n * 100, test_acc / test_n * 100)
        
        if args.optimizer =='swa'  and (e + 1) >= args.swa_start and args.eval_freq>1:
            if e == 0 or e % args.eval_freq == args.eval_freq - 1 or e == args.epochs - 1:
                opt.swap_swa_sgd()
                opt.bn_update(train_loaders, model, device='cuda')
                #swa_res = utils.eval_epoch(test_loaders['test'], model)
                opt.swap_swa_sgd()

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
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument('--swa_start', type=float, default=161)
    parser.add_argument('--swa_lr', type=float, default=0.025)
    parser.add_argument('--final_lr', type =int, default=0.1)
    parser.add_argument('--gamma', type=float, default=1e-3)
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam coefficients beta_1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam coefficients beta_2')
    #BatchEVAL
    parser.add_argument('--eval_freq', type= int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()
    main(args)
