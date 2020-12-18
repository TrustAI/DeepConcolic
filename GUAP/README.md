# GUAP

Tool for generating spatial-transfermed or additive universarial perturbations, the paper '[Generalizing Universal Adversarial Attacks Beyond Additive Perturbations](https://arxiv.org/pdf/2010.07788.pdf)' was accepted by [ICDM 2020](http://icdm2020.bigke.org/).

Please cite Yanghao Zhang, Wenjie Ruan, Fu Wang, and Xiaowei Huang, Generalizing Universal Adversarial Attacks Beyond Additive Perturbations, The IEEE International Conference on Data Mining (ICDM 2020), November 17-20, 2020, Sorrento, Italy

![overview](/savefig/overview.png "overview")

In this paper, for the first time we propose a unified and flexible framework, which can capture the distribution of the unknown additive and non-additive adversarial perturbations jointly for crafting Generalized Universal Adversarial Perturbations. 
Specifically, GUAP can generate either additive (i.e., l_inf-bounded) or non-additive (i.e., spatial transformation) perturbations, or a com- bination of both, which considerably generalizes the attacking capability of current universal attack methods.


## Running environment:
python 3.6.10

pytorch 1.5.0

## Colab demo:

There is also a notebook demo ```Colab_GUAP.ipynb```, which can be run on the Colab.

## Generalizing UAP for Cifar10:
```
	python run_cifar --gpuid 0 --model VGG19
```
## Generalizing UAP for ImageNet:
```
	python run_imagenet.py --gpuid 0,1 --model ResNet152
```

## Experimental results:

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/savefig/Cifar10.png" width="70%">

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/savefig/ImageNet.png" width="71%">


