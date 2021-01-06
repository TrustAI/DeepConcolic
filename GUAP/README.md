# GUAP

Tool for generating spatial-transfermed or additive universarial perturbations, the paper '[Generalizing Universal Adversarial Attacks Beyond Additive Perturbations](https://arxiv.org/pdf/2010.07788.pdf)' was accepted by [ICDM 2020](http://icdm2020.bigke.org/).

Please cite Yanghao Zhang, Wenjie Ruan, Fu Wang, and Xiaowei Huang, Generalizing Universal Adversarial Attacks Beyond Additive Perturbations, The IEEE International Conference on Data Mining (ICDM 2020), November 17-20, 2020, Sorrento, Italy

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/figs/workflow.png" width="100%">


In this paper, for the first time we propose a unified and flexible framework, which can capture the distribution of the unknown additive and non-additive adversarial perturbations jointly for crafting Generalized Universal Adversarial Perturbations. 
Specifically, GUAP can generate either additive (i.e., l_inf-bounded) or non-additive (i.e., spatial transformation) perturbations, or a combination of both, which considerably generalizes the attacking capability of current universal attack methods.


## Colab demo:

There is also a notebook demo [```Colab_GUAP.ipynb```](https://nbviewer.jupyter.org/github/YanghaoZYH/GUAP/blob/master/Colab_GUAP.ipynb), which can be run on the Colab.


## Running environment:

```
pip install torch torchvision matplotlib
```

## Download target Models
```
cd saved_models
wget -O cifar10_vgg19.pth https://www.dropbox.com/s/qkl5224tbo9flvg/cifar10_vgg19.pth?dl=0
wget -O cifar10_resnet101.pth https://www.dropbox.com/s/1x8yj9u7f8zlff6/cifar10_resnet101.pth?dl=0
wget -O cifar10_dense121.pth https://www.dropbox.com/s/pbq408117fd7935/cifar10_dense121.pth?dl=0
wget -O fashion_mnist_modela.pth https://www.dropbox.com/s/otjg9tusqogxodf/fashion_mnist_modela.pth?dl=0
cd ..
```

## Command to Run 
```
usage: run_fashion_mnist.py [-h] [--dataset DATASET] [--lr LR]
                            [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                            [--l2reg L2REG] [--beta1 BETA1] [--tau TAU]
                            [--eps EPS] [--model MODEL]
                            [--manualSeed MANUALSEED] [--gpuid GPUID] [--cuda]
                            [--resume] [--outdir OUTDIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Fashion-MNIST
  --lr LR               Learning rate
  --batch-size BATCH_SIZE
  --epochs EPOCHS       number of epochs to train for
  --l2reg L2REG         weight factor for l2 regularization
  --beta1 BETA1         beta1 for adam. default=0.5
  --tau TAU             max flow magnitude, default=0.1
  --eps EPS             allow for linf noise. default=0.1
  --model MODEL         modelA
  --manualSeed MANUALSEED
                        manual seed
  --gpuid GPUID         multi gpuid
  --cuda                enables cuda
  --resume              load pretrained model
  --outdir OUTDIR       output dir
```

## Generalizing UAP for Fashion_MNIST:
```
python run_fashion_mnist.py --cuda --gpuid 0 --resume
```

## Generalizing UAP for Cifar10:
```
python run_cifar.py --cuda --gpuid 0 --model VGG19
```
## Generalizing UAP for ImageNet:
```
python run_imagenet.py --cuda --gpuid 0,1 --model ResNet152
```

## Experimental results:

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/figs/Cifar10.png" width="70%">

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/figs/ImageNet.png" width="71%">


## GUAP Demonstration
Available at http://guap.yanghaozhang.com/
