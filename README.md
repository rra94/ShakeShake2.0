# Shake-Shake Regularization with cutout and adabound optimizer

## Methodology choices

### State of the Art (as of Apr'19)

### Cutout

## Error-Rate

### CIFAR-100
|Model|This implementaion |Shake-shake|Shake-shake with cutout|
|:---:|:---:|:---:|:---:|
|ResNeXt29-2x4x64d | TODO |15.58 | 15.20|

## Train ResNet29-2x64d  with cutout size 8 and SGD optimizer for CIFAR-100 
```
python train.py --label 100 --depth 29 --w_base 64 --lr 0.025 --epochs 1800 --batch_size 128  --half_length=8 --optimizer='sdg'
```
## References

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

This code is built over [this repo](https://github.com/owruby/shake-shake_pytorch)
