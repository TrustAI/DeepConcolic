# DeepConcolic (Concolic Testing for Deep Neural Networks)

![alt text](images/deepconcolic-logo2.png)


Concolic testing alternates between CONCrete program execution and symbOLIC analysis to explore the execution paths of a software program and to increase code coverage. In this paper, we develop the first concolic testing approach for Deep Neural Networks (DNNs). More specifically, we utilise quantified linear arithmetic over rationals to express test requirements that have been studied in the literature, and then develop a coherent method to perform concolic testing with the aim of better coverage. Our experimental results show the effectiveness of the concolic testing approach in both achieving high coverage and finding adversarial examples.

The paper is available in https://arxiv.org/abs/1805.00089.

# Work Flow
![alt text](ASE-experiments/PaperData/Work_Flow.png)

# Sample Results
![alt text](ASE-experiments/PaperData/Adversarial_Examples-b.png)
![alt text](ASE-experiments/PaperData/Concolic_Testing_Results.png )

# Run  

```
usage: deepconcolic.py [-h] [--model MODEL] [--inputs DIR] [--outputs DIR]
                       [--training-data DIR] [--criterion nc, ssc...]
                       [--labels FILE] [--mnist-dataset] [--cifar10-dataset]
                       [--vgg16-model] [--norm linf, l0] [--input-rows INT]
                       [--input-cols INT] [--input-channels INT]
                       [--cond-ratio FLOAT] [--top-classes INT]
                       [--layer-index INT]

Concolic testing for neural networks

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the input neural network model (.h5)
  --inputs DIR          the input test data directory
  --outputs DIR         the outputput test data directory
  --training-data DIR   the extra training dataset
  --criterion nc, ssc...
                        the test criterion
  --labels FILE         the default labels
  --mnist-dataset       MNIST dataset
  --cifar10-dataset     CIFAR-10 dataset
  --vgg16-model         vgg16 model
  --norm linf, l0       the norm metric
  --input-rows INT      input rows
  --input-cols INT      input cols
  --input-channels INT  input channels
  --cond-ratio FLOAT    the condition feature size parameter (0, 1]
  --top-classes INT     check the top-xx classifications
  --layer-index INT     to test a particular layer
```

The neural network model under tested is specified by ``--model`` and a set of raw test data should be given
by using ``--inputs``. Some popular datasets like MNIST and CIFAR10 can be directly specified by using
``--mnist-dataset`` and ``--cifar10-dataset`` directly. ``--criterion`` is used to choose the coverage 
criterion and ``--norm`` helps select the norm metric to measure the distance between inputs. Some examples
to run DeepConcolic are in the following.

To run an MNIST model

```
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --mnist-data --outputs outs/
```

To run an CIFAR10 model

```
python deepconcolic.py --model ../saved_models/cifar10_complicated.h5 --cifar10-data --outputs outs/
```

To test a particular layer
```
python deepconcolic.py --model ../saved_models/cifar10_complicated.h5 --cifar10-data --outputs outs/ --layer-index 2
```

To run MC/DC for DNNs on the CIFAR-10 model

```
python deepconcolic.py --model ../saved_models/cifar10_complicated.h5 --criterion ssc --cond-ratio 0.1 --cifar10-data --outputs outs
```

To run MC/DC for DNNs on the VGG16 model

```
python  deepconcolic.py --vgg16-model --inputs data/ --outputs outs --cond-ratio 0.1 --top-classes 5 --labels labels.txt --criterion ssc
```

To run Concolic Sign-sign-coverage (MC/DC) for DNNs on the MNIST model

```
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --criterion ssclp --mnist-data --outputs outs
```

DeepConcolic nows supports an experimental fuzzing engine. Try ``--fuzzing`` to use it. The following command will result in: one ``mutants`` folder, one ``advs`` folder for adversarial examples and an adversarial list ``adv.list``.

```
python src/deepconcolic.py --fuzzing --model ./saved_models/mnist2.h5 --inputs data/mnist-seeds/ --outputs outs --input-rows 28 --input-cols 28
```

### Concolic Testing on Lipschitz Constants for DNNs

To run Lipschitz Constant Testing, please refer to instructions in folder "Lipschitz Constant Testing".

# Dependencies
We suggest to create an environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html) as follows:
```
conda create --name deepconcolic
conda activate deepconcolic
conda install opencv 
pip3 install tensorflow\>=2.3
pip3 install pulp\>=2
pip3 install adversarial-robustness-toolbox\>=1.3
```

Note as of September 2020 one may need to append `--use-feature=2020-resolver` at the end of each `pip3 install` command-line to work-around errors in dependency resolution.  Further missing dependency errors for a package _p_ can then be solved by uninstalling/installing _p_.

# Publications

```
@inproceedings{swrhkk2018,
  AUTHOR    = { Sun, Youcheng
                and Wu, Min
                and Ruan, Wenjie
                and Huang, Xiaowei
                and Kwiatkowska, Marta
                and Kroening, Daniel },
  TITLE     = { Concolic Testing for Deep Neural Networks },
  BOOKTITLE = { Automated Software Engineering (ASE) },
  PUBLISHER = { ACM },
  PAGES     = { 109--119 },
  ISBN      = { 978-1-4503-5937-5 },
  YEAR      = { 2018 }
}
```
```
@article{sun2018testing,
  AUTHOR    = { Sun, Youcheng
                and Huang, Xiaowei
                and Kroening, Daniel },
  TITLE     = { Testing Deep Neural Networks },
  JOURNAL   = { arXiv preprint arXiv:1803.04792 },
  YEAR      = { 2018 }
}
```
