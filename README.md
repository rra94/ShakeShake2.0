# Shake-Shake Regularization with Cutout and Optimizer

## Task

Achieve State of the ART(SOTA) on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). The inspiration for the methodology comes from [this](https://paperswithcode.com/sota/image-classification-on-cifar-100) table.

### Dataset details

CIFAR-100 dataset has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs). 

We perform 100 class-classification. 


## Methodology choices

Even though there are models with higher accuracy like [GPIPE](https://arxiv.org/pdf/1811.06965v4.pdf) and [Autoagumet](https://arxiv.org/pdf/1805.09501.pdf). These models are not replicable easily in real life. For instance, AutoAgument used 5000 hours of GPU time to train on CIFAR-10 then uses the same polies on CIFAR 100. On the other had [GPIPE](https://arxiv.org/pdf/1811.06965v4.pdf) useas a vast ameoba net which is trained on [ImageNet](http://www.image-net.org/) and then performs transfer learning. Thus, we go with the next best set of methods nameley ResNeXt with Shake-Shake [3]. We also choose newer optimizers for faster convergence [4,5]. 

Here we discuss the design choices for our experiments. 

### Architecture

* ResNeXt [1]:  We use the ResNeXt29-2x4x64d(The network has a depth of 29,  2 residual branches with 4 grouped convolutions and the first residualblock has a width of 64). We use the same model as [2] to replicate their results.

### Regularization 

Shake Shake [3]: This is a new regularization technique aimed at helping deep learning practitioners faced with an overfit problem. The idea is to replace, in a multi-branch network, the standard summation of parallel branches with a stochastic affine combination.

"Shake" means that all scaling coefficients are overwritten with new random numbers before the pass.  "Even" means that all scaling coefficients are set to 0.5 before the pass.  "Keep" means that we keep, for the backward pass, the scaling coefficients used during the forward pass.  "Batch" means that, for each residual block_i, we apply the same scaling coefficientfor all the images in the mini-batch.   "Image" means that,  for each residual block_i,  we apply a different scaling coefficient for each image in the mini-batch.

Specifically we use the Shake-Shake-Image (SSI) regularization i.e. "Shake" for both forward and backward passes and the level is set to "Image". By level we mean: Letx_0 denote the original input mini-batch tensor of dimensions 128x3x32x32. The first dimension « stacks » 128 images of dimensions 3x32x32. Inside the secondstage of a 26 2x32d model, this tensor is transformed into a mini-batch tensorxiof dimensions 128x64x16x16. Applying Shake-Shake regularization at the Image level means slicing this tensor along the first dimension and, for each of the 128 slices, multiplying the j slice (of dimensions 64x16x16) with a scalar $α_{i,j}$(or $(1−α)_{i,j}$).

### Weight Initialization

* Random weight initializations are used. 

### Data Agumentation

Apart from horizontal flip and random crop we perfom the following data agumentations as well:

* Cutout [2]: A small, randomly selected patch(s) of the image is masked for each image before it is used for training. The authors claim that the cutout technique simulates occluded examples and encourages the model to take more minor features into consideration when making decisions, rather than relying on the presence of a few major features.

### Otimizers

We consider the following optimizers:

* SGD: We first use the standard stochastic gradient decent.

* Adabound [4]: AdaBound is an optimizer that behaves like Adam at the beginning of training, and gradually transforms to SGD at the end. The ``final_lr`` parameter indicates AdaBound would transforms to an SGD with this learning rate. It is not very sensitive to its hyperparameters. 

* SWA [5]:  The key idea of SWA is to average multiple samples produced by SGD with a modified learning rate schedule. We use a cyclical learning rate schedule that causes SGD to explore the set of points in the weight space corresponding to high-performing networks. The authors claim that SWA converges more quickly than SGD, and to wider optima that provide higher test accuracy.

### Normalization

Each branch has batch nromalization.

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


Our method achieves results comparable to the author' implementation in.

### Discussion


## Future Steps

* Replace BatchNorm with [Fixup initalization](https://arxiv.org/abs/1901.09321)

* Try [dropblock](https://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks.pdf) instead of cutout

* Try (PyramidNet+ShakeDrop)[https://arxiv.org/pdf/1802.02375.pdf]

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


1. [RESNEXT](https://github.com/facebookresearch/ResNeXt)
```
@inproceedings{xie2017aggregated,
  title={Aggregated residual transformations for deep neural networks},
  author={Xie, Saining and Girshick, Ross and Doll{\'a}r, Piotr and Tu, Zhuowen and He, Kaiming},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1492--1500},
  year={2017}
}

```

2. [Improved Regularization of Convolutional Neural Networks with Cutout.](https://github.com/uoguelph-mlrg/Cutout).
```
@article{devries2017cutout,  
  title={Improved Regularization of Convolutional Neural Networks with Cutout},  
  author={DeVries, Terrance and Taylor, Graham W},  
  journal={arXiv preprint arXiv:1708.04552},  
  year={2017}  
}
```
3. [Shake-shake regularization](https://github.com/xgastaldi/shake-shake).
```
@article{gastaldi2017shake,
  title={Shake-shake regularization},
  author={Gastaldi, Xavier},
  journal={arXiv preprint arXiv:1705.07485},
  year={2017}
}
```

4. [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://github.com/Luolc/AdaBound).

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

5. [Averaging Weights Leads to Wider Optima and Better Generalization](https://github.com/izmailovpavel/contrib_swa_examples)
```
@article{izmailov2018averaging,
  title={Averaging Weights Leads to Wider Optima and Better Generalization},
  author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:1803.05407},
  year={2018}
}
```

This code is built over [this repo](https://github.com/owruby/shake-shake_pytorch)
