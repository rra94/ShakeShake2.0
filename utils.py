# -*- coding: utf-8 -*  
import torch
from torch.autograd import Variable
import os
import math
import json
from datetime import datetime
import torch 
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr


def prepare_pt_context(num_gpus,batch_size):
    use_cuda = (num_gpus > 0)
    batch_size *= max(1, num_gpus)
    return use_cuda, batch_size

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr



def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return acc


class cutout(object):
    def __init__(self, half_length,  nholes):
        self.n_holes = nholes
        self.length = half_length
    def __call__(self, img):
        if self.n_holes ==0:
            #print("no cutout")
            return img

        h,w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def eval_epoch(test_loader, model):
    loss_func= nn.CrossEntropyLoss().cuda()
    model.eval()
    test_loss =0.0
    test_acc =0.0
    test_n =0.0

    for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
        #if criterion=='swa':

        with torch.no_grad():
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            test_loss += loss.item() * t.size(0)
            test_acc += accuracy(y, t).item()
            test_n += t.size(0)
    return (test_loss, test_acc, test_n) 

class Logger:

    def __init__(self, log_dir, headers, mod):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, str(mod)+"log.txt"), "w")
        header_str = "\t".join(headers + ["EndTime."])
        self.print_str = "\t".join(["{}"] + ["{:.6f}"] * (len(headers) - 1) + ["{}"])

        self.f.write(header_str + "\n")
        self.f.flush()
        print(header_str)

    def write(self, *args):
        now_time = datetime.now().strftime("%m/%d %H:%M:%S")
        self.f.write(self.print_str.format(*args, now_time) + "\n")
        self.f.flush()
        print(self.print_str.format(*args, now_time))

    def write_hp(self, hp):
        json.dump(hp, open(os.path.join(self.log_dir, "hp.json"), "w"))

    def close(self):
        self.f.close()
