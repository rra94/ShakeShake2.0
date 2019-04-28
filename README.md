# Shake-Shake Regularization with Cutout and Optimizer

## Task

Achieve State of the ART(SOTA) on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). The inspiration comes from [this](https://paperswithcode.com/sota/image-classification-on-cifar-100) table.

### Dataset details

This dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs). 

We perform 100 class-classification. 


## Methodology choices

Here we discuss the design choices for our experiments. 

### Model

* ResNeXt:  We use the ResNeXt29-2x4x64d ().

### Weight Initialization

* Random 

### Data Agumentation

* 
* Cutout 

### Otimizers

* SGD
* Adabound
* SWA

### Normalization

### Dropout


## Results

All experiments are run on one NVIDIA RTX2080 Ti. 

### CIFAR-100 (Best shot Error Rate)

|Model|This implementaion |Epochs |Paper|
|:---:|:---:|:---:|:---:|
|ResNeXt29-2x4x64d | TODO | |16.56 |
|ResNeXt29-2x4x64d + shakeshake | TODO | |15.58 |
|ResNeXt29-2x4x64d + shakeshake+  cutout + SDG| TODO | 1800| 15.20|
|ResNeXt29-2x4x64d + shakeshake + cutout + ADABOUND| TODO  | 1800 |NA|
|ResNeXt29-2x4x64d + shakeshake + cutout + SWA| TODO | |NA|
|State of the Art([GPIPE](https://arxiv.org/pdf/1811.06965v4.pdf)) |  - | - | 9.43|


Our method achieves results comparable to 

## Future Steps

* Replace BatchNorm with Fixup initalization

* Try dropblock instead of cutout

* Try pyramidnet+shakeDrop

## Train ResNet29-2x64d  with cutout size 8 and SGD optimizer for CIFAR-100 

### Dependencies

* Pytorch 1.0.1
* Python 3.6+ 
* Install all from requirements.txt

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --label 100 --depth 29 --w_base 64 --lr 0.025 --epochs 1800 --batch_size 128  --half_length=8 --optimizer='sdg'
```

### Execution options

* This code has parallel capabilites, use `CUDA_VISIBLE` to add devices
* Switch optimizers with `--optimizer`. Available SGD, ADABOUND, SWA
* set cutout with `--half_length`
* added capability for many cutouts with `--nholes`. Set to 0 for no cutout.
* batch evaluation with `--eval_freq`
* switch cifar 10 and 100 with `-label 100`
* set learning rate `--lr` (intial learning rates for SWA, ADABOUND)
* set epochs with `-epochs
* set momentum with `--momentum`
* Change depth of resnet `--depth`
* change weight decay `--weight_decay`
* set batch size `--batch_size`


### SWA Functionality
* set swa learning rate `--swa_lr`
* set start epoch for swa `-swa_start`


## References


[RESNEXT](https://github.com/facebookresearch/ResNeXt)
```
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}

```

[Improved Regularization of Convolutional Neural Networks with Cutout.](https://github.com/uoguelph-mlrg/Cutout).
```
@article{devries2017cutout,  
  title={Improved Regularization of Convolutional Neural Networks with Cutout},  
  author={DeVries, Terrance and Taylor, Graham W},  
  journal={arXiv preprint arXiv:1708.04552},  
  year={2017}  
}
```
[Shake-shake regularization](https://github.com/xgastaldi/shake-shake).
```
@article{gastaldi2017shake,
  title={Shake-shake regularization},
  author={Gastaldi, Xavier},
  journal={arXiv preprint arXiv:1705.07485},
  year={2017}
}
```

[Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://github.com/Luolc/AdaBound).

```text
@inproceedings{Luo2019AdaBound,
  author = {Luo, Liangchen and Xiong, Yuanhao and Liu, Yan and Sun, Xu},
  title = {Adaptive Gradient Methods with Dynamic Bound of Learning Rate},
  booktitle = {Proceedings of the 7th International Conference on Learning Representations},
  month = {May},
  year = {2019},
  address = {New Orleans, Louisiana}
}
```

[Averaging Weights Leads to Wider Optima and Better Generalization](https://github.com/izmailovpavel/contrib_swa_examples)
```
@article{izmailov2018averaging,
  title={Averaging Weights Leads to Wider Optima and Better Generalization},
  author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:1803.05407},
  year={2018}
}
```

This code is built over [this repo](https://github.com/owruby/shake-shake_pytorch)
