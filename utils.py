# -*- coding: utf-8 -*

import os
import math
import json
from datetime import datetime
import torch 
import numpy as np


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

class Logger:

    def __init__(self, log_dir, headers):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, "log.txt"), "w")
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
