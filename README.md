# DeepConcolic (Testing for Deep Neural Networks)


![alt text](images/deepconcolic-logo2.png)


# General Introduction

This repository includes a few software packages, all of which are dedicated for the analysis of deep neural netowrks (or tree ensembles) over its safety and/or security properties. 
1. DeepConcolic, a coverage-guided testing tool for convolutional neural networks. Now, it includes a major upgrade based on Bayesian Network based Abstraction. 
2. testRNN, a coverage-guided testing tool for Long short-term memory models (LSTMs). LSTMs are a major class of recurrent neural networks. 
3. EKiML, a tool for backdoor embedding and detection for tree ensembles.
4. GUAP: a generalised universal adversarial perturbation. It generates universersal adversarial perburbation that may be applied to many inputs at the same time. 

In the following, after the installation and download of example models, we will present them one by one.  

# Installation

First of all, please set up a conda environment

```
conda create --name deepconcolic python==3.7
conda activate deepconcolic
```
This should be followed by installing software dependencies:
```
conda install opencv nltk matplotlib
pip3 install tabulate scikit-learn tensorflow==2.3.0 pulp keract np_utils adversarial-robustness-toolbox pomegranate==0.13.4 scipy numpy pysmt saxpy keras scikit-image menpo patool --use-feature=2020-resolver
```
# Download Example Models
We use Fashion-MNIST dataset as the running example. The following are two pre-trained mmodels, one larger and one smaller.  
```
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/small_model_fashion_mnist.h5
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/large_model_fashion_mnist.h5
```

# Tool 1 -- DeepConcolic: Concolic Testing for Convolutional Neural Networks 

Concolic testing alternates between CONCrete program execution and symbOLIC analysis to explore the execution paths of a software program and to increase code coverage. In this paper, we develop the first concolic testing approach for Deep Neural Networks (DNNs). More specifically, we utilise quantified linear arithmetic over rationals to express test requirements that have been studied in the literature, and then develop a coherent method to perform concolic testing with the aim of better coverage. Our experimental results show the effectiveness of the concolic testing approach in both achieving high coverage and finding adversarial examples.

The paper is available at https://arxiv.org/abs/1805.00089.

In the following, we first present the original ASE2018 version, and then introduce two new upgrades (fuzzing engine and Bayesian network based abstraction). 

## ASE2018 Version

### Work Flow
![alt text](ASE-experiments/PaperData/Work_Flow.png)

### Sample Results
![alt text](ASE-experiments/PaperData/Adversarial_Examples-b.png)
![alt text](ASE-experiments/PaperData/Concolic_Testing_Results.png )

### Command to Run  

```
usage: deepconcolic.py [-h] [--model MODEL] [--vgg16-model] [--inputs DIR]
                       --outputs DIR [--criterion nc, ssc...] [--setup-only]
                       [--init INT] [--max-iterations INT] [--save-all-tests]
                       [--rng-seed SEED] [--labels FILE]
                       [--dataset {OpenML:har,cifar10,fashion_mnist,mnist}]
                       [--extra-tests DIR [DIR ...]] [--filters {LOF}]
                       [--norm {linf,l0}] [--norm-factor FLOAT]
                       [--lb-hard FLOAT] [--lb-noise FLOAT] [--input-rows INT]
                       [--input-cols INT] [--input-channels INT]
                       [--mcdc-cond-ratio FLOAT] [--top-classes CLS]
                       [--layers LAYER [LAYER ...]] [--feature-index INT]
                       [--fuzzing] [--num-tests INT] [--num-processes INT]
                       [--sleep-time INT] [--dbnc-spec SPEC]
                       [--dbnc-abstr PKL]

Concolic testing for neural networks

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the input neural network model (.h5)
  --vgg16-model         use keras's default VGG16 model (ImageNet)
  --inputs DIR          the input test data directory
  --outputs DIR         the outputput test data directory
  --criterion nc, ssc...
                        the test criterion
  --setup-only          only setup the coverage critierion and analyzer, and
                        terminate before engine initialization and startup
  --init INT            number of test samples to initialize the engine
  --max-iterations INT  maximum number of engine iterations (use < 0 for
                        unlimited)
  --save-all-tests      save all generated tests in output directory; only
                        adversarial examples are kept by default
  --rng-seed SEED       Integer seed for initializing the internal random
                        number generator, and therefore get some(what)
                        reproducible results
  --labels FILE         the default labels
  --dataset {OpenML:har,cifar10,fashion_mnist,mnist}
                        selected dataset
  --extra-tests DIR [DIR ...]
                        additonal directories of test images
  --filters {LOF}       additional filters used to put aside generated test
                        inputs that are too far from training data (there is
                        only one filter to choose from for now; the plural is
                        used for future-proofing)
  --norm {linf,l0}      the norm metric
  --norm-factor FLOAT   norm distance upper threshold above which generated
                        inputs are rejected by the oracle (default is 1/4)
  --lb-hard FLOAT       hard lower bound for the distance between original and
                        generated inputs (concolic engine only---default is
                        1/255 for image datasets, 1/100 otherwise)
  --lb-noise FLOAT      extra noise on the lower bound for the distance
                        between original and generated inputs (concolic engine
                        only---default is 1/10)
  --input-rows INT      input rows
  --input-cols INT      input cols
  --input-channels INT  input channels
  --mcdc-cond-ratio FLOAT
                        the condition feature size parameter (0, 1]
  --top-classes CLS     check the top-CLS classifications for models that
                        output estimations for each class (e.g. VGG*)
  --layers LAYER [LAYER ...]
                        test layers given by name or index
  --feature-index INT   to test a particular feature map
  --fuzzing             to start fuzzing
  --num-tests INT       number of tests to generate
  --num-processes INT   number of processes to use
  --sleep-time INT      fuzzing sleep time
  --dbnc-spec SPEC      Feature extraction and discretisation specification
  --dbnc-abstr PKL, --bn-abstr PKL
                        input BN abstraction (.pkl)
```

The neural network model under tested is specified by ``--model`` and a set of raw test data should be given
by using ``--inputs``. Some popular datasets like MNIST and CIFAR10 can be directly specified by using the
``--dataset`` option directly. ``--criterion`` is used to choose the coverage
criterion and ``--norm`` helps select the norm metric to measure the distance between inputs. Some examples
to run DeepConcolic are in the following.

To run an MNIST model

```
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --dataset mnist --outputs outs/
```

To run an CIFAR10 model

```
python -m deepconcolic.main --model saved_models/cifar10_complicated.h5 --dataset cifar10 --outputs outs/
```

To test a particular layer
```
python -m deepconcolic.main --model saved_models/cifar10_complicated.h5 --dataset cifar10 --outputs outs/ --layers 2
```

To run MC/DC for DNNs on the CIFAR-10 model

```
python -m deepconcolic.main --model saved_models/cifar10_complicated.h5 --criterion ssc --mcdc-cond-ratio 0.1 --dataset cifar10 --outputs outs
```

To run MC/DC for DNNs on the VGG16 model (with input images from the ``data`` sub-directory)

```
python -m deepconcolic.main --vgg16-model --inputs data/ --outputs outs --mcdc-cond-ratio 0.1 --top-classes 5 --labels labels.txt --criterion ssc
```

To run Concolic Sign-sign-coverage (MC/DC) for DNNs on the MNIST model

```
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --dataset mnist --outputs outs --criterion ssclp
```

## Fuzzing Engine

DeepConcolic nows supports an experimental fuzzing engine. Try ``--fuzzing`` to use it. The following command will result in: one ``mutants`` folder, one ``advs`` folder for adversarial examples and an adversarial list ``adv.list``.

```
python -m deepconcolic.main --fuzzing --model saved_models/large_model_fashion_mnist.h5 --num-processes 2 --inputs data/mnist-seeds/ --outputs outs --input-rows 28 --input-cols 28
```

## Bayesian Network based Abstraction

To run Concolic BN-based Feature coverage (BFCov) for DNNs on the MNIST model
```
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --criterion bfc --norm linf --dataset mnist --outputs outs --dbnc-spec dbnc/example.yaml
```
See [the example YAML specification](dbnc/example.yaml) for details on how to configure the BN-based abstraction.


To run Concolic BN-based Feature-dependence coverage (BFdCov) for DNNs on the MNIST model
```
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --criterion bfdc --norm linf --dataset mnist --outputs outs --dbnc-spec dbnc/example.yaml
```
You could adjust the following two parameters in the DBNC specification file defined by `--dbnc-spec` to dump the generated bayesian network to files `bn4trained.yml` and `bn4tests.yml`. 
 ```  
    dump_bn_with_trained_dataset_distribution: True,
    dump_bn_with_final_dataset_distribution: True,
 ```


# Tool 2 -- testRNN: Coverage Guided Testing for Recurrent Nueral Networks

For long short-term memory models (LSMTs), we design new coverage metrics to consider the internal behaviour of the LSTM layers in processing sequential inputs. We consider not only the tighter metric that quantifies the temporal behaviour (i.e., temporal coverage) but also some looser metrics that quantify either the gate values (i.e., Neuron Coverage and Boundary Coverage) or value change in one step (i.e., Stepwise Coverage). 

The paper is available at https://arxiv.org/pdf/1911.01952.pdf.
       
#### Four coverage test metrics are applicable: 
1. Neuron Coverage (NC), 
2. Boundary Coverage (BC), 
3. Stepwise Coverage (SC), 
4. Temporal Coverage (TC)

#### A few pre-trained LSTM models: 
1. Fashion-MNIST
2. Sentiment Analysis, 
3. MNIST Handwritten Digits, 
4. UCF101 (need to download and put into the dataset file)

As running example, we download the pre-trained Fasion-MNIST model as follows. 

```
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/fashion_mnist_lstm.h5
```

## Command to Run: 

We have two commands to run testing procedure and to run result analysis procedure, respectively. 

#### to run testing procedure
```
python -m testRNN.main --model <modelName> 
                           --TestCaseNum <Num. of Test Cases> 
                           --threshold_SC <SC threshold> 
                           --threshold_BC <BC threshold> 
                           --symbols_TC <Num. of symbols> 
                           --seq <seq in cells to test>
                           --mode <modeName>
                           --outputs <output directory>
```
where 
1. \<modelName> is in {sentiment, mnist, fashion_mnist, ucf101}
2. \<Num. of Test Cases> is the expected number of test cases
3. \<Mutation Method> is in {'random', 'genetic'}
4. \<SC threshold> is in [0, 1]  
5. \<BC threshold> is in [0, 1]
6. \<Num. of symbols> is in {1, 2, 3...}
7. \<seq in cells to test> is in {mnist: [4, 24], fashion_mnist: [4, 24], sentiment: [400, 499], ucf101: [0, 10]}
8. \<modeName> is in {train, test} with default value test 
9. \<output directory> specifies the path of the directory to save the output record and generated examples

For example, we can run the following 
```
python -m testRNN.main --model fashion_mnist --TestCaseNum 10000 --Mutation random --threshold_SC 0.6 --threshold_BC 0.7 --symbols_TC 3 --seq [4,24] --outputs testRNN_output
```
which says that, we are working with Fashion-MNIST model, and the genetic algorithm based test case generation will terminate when the number of test cases is over 10000. We need to specify other parameters including threshold_SC, threshold_BC, symbols_TC, and seq. Moreover, the log is generated to the file testRNN_output/record.txt. Also the output of adversarial examples can be found in testRNN_output/adv_output
    
# Tool 3 -- EKiML: Embedding Knolwedge into Tree Ensembles

In this tool, we consider embedding knowledge into machine learning models. The knowledge expression we considered can express e.g., robustness and resilience to backdoor attack, etc. That is, we can "embed" knowledge into a tree ensemble, representing a backdoor attack on the tree ensemble. Also, we can "detect" if a tree ensemble has been attacked. 

The paper is available at https://arxiv.org/pdf/2010.08281.pdf.

## Download pre-trained models

As the running example, we download the pre-trained HAR tree model as follows. 

```
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/har_tree_black-box.npy
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/har_forest_black-box.npy
```

## Command to Run

```
python -m EKiML.main --Dataset <DatasetName> 
		     --Mode <modeName>
		     --Embedding_Method <embeddingMethod>
		     --Model <modeType>
		     --Pruning <pruningFlag>
		     --SaveModel <saveModelFlag>
		     --workdir <workDirectory>
```
where the flags have multiple options: 

1. \<DatasetName> is in {'iris', 'har', 'breast_cancer', 'mushroom', 'nursery, 'cod-rna', 'sensorless', 'mnist'}.
2. \<modeName> is in {'embedding', 'synthesis'}, where 'synthesis' denotes the "extraction". 
3. \<embeddingMethod> is in {'black-box', 'white-box'}
4. \<modeType> is in {'forest', 'tree'}
5. \<pruningFlag> is in {True, False}, with default value False
6. \<saveModelFlag> is in {True, False}, with default value False
7. \<workDirectory> is the working directory, with default value 'EKiML_workdir'
8. \<Datadir> is the directory where dataset files are located (default is 'EKiML/dataset')

For example, we can run the following
```
python -m EKiML.main --Dataset har --Mode synthesis --Embedding_Method black-box --Model tree --workdir 'EKiML_har' --Datadir 'datasets'
```
which suggests that we are considering the HAR dataset, tryng to synthesise knowledge from a pre-trained tree by applying our black-box synthesis algorithm.


# Tool 4 -- GUAP: Generalised Universal Adversarial Perturbation 

Tool for generating spatial-transfermed or additive universarial perturbations, the paper '[Generalizing Universal Adversarial Attacks Beyond Additive Perturbations](https://arxiv.org/pdf/2010.07788.pdf)' was accepted by [ICDM 2020](http://icdm2020.bigke.org/).

Please cite Yanghao Zhang, Wenjie Ruan, Fu Wang, and Xiaowei Huang, Generalizing Universal Adversarial Attacks Beyond Additive Perturbations, The IEEE International Conference on Data Mining (ICDM 2020), November 17-20, 2020, Sorrento, Italy

The paper is avaiable at: https://arxiv.org/pdf/2010.07788.pdf 

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
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/cifar10_vgg19.pth
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/cifar10_resnet101.pth 
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/cifar10_dense121.pth 
wget -P saved_models https://cgi.csc.liv.ac.uk/~acps/models/fashion_mnist_modela.pth
```

## Command to Run
(from within the ```GUAP``` sub-directory)
```
usage: run_xxxxxx.py [-h] [--dataset DATASET] [--lr LR]
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
  --beta1 BETA1         beta1 for adam
  --tau TAU             max flow magnitude
  --eps EPS             allow for linf noise
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
python run_cifar.py --cuda --gpuid 0 --model VGG19 --tau 0.1 --eps 0.03
```
## Generalizing UAP for ImageNet:
```
python run_imagenet.py --cuda --gpuid 0,1 --model ResNet152 --tau 0.1 --eps 0.03
```

## Experimental results:

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/figs/Cifar10.png" width="70%">

<img src="https://github.com/YanghaoZYH/GUAP/blob/master/figs/ImageNet.png" width="71%">






# Publications

### For DeepConcolic,
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
```
@article{10.1145/3358233, 
author = {Sun, Youcheng and Huang, Xiaowei and Kroening, Daniel and Sharp, James and Hill, Matthew and Ashmore, Rob}, 
title = {Structural Test Coverage Criteria for Deep Neural Networks}, 
year = {2019}, 
issue_date = {October 2019}, 
publisher = {Association for Computing Machinery}, 
address = {New York, NY, USA}, 
volume = {18}, 
number = {5s}, 
issn = {1539-9087}, 
url = {https://doi.org/10.1145/3358233}, 
doi = {10.1145/3358233}, 
journal = {ACM Trans. Embed. Comput. Syst.}, 
articleno = {Article 94}, 
numpages = {23}, 
keywords = {test criteria, Neural networks, test case generation} }
```

### For testRNN, 
```
@article{DBLP:journals/corr/abs-1911-01952,
  author    = {Wei Huang and
               Youcheng Sun and
               James Sharp and
               Xiaowei Huang},
  title     = {Test Metrics for Recurrent Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1911.01952},
  year      = {2019},
  url       = {http://arxiv.org/abs/1911.01952},
  archivePrefix = {arXiv},
  eprint    = {1911.01952},
  timestamp = {Thu, 05 Mar 2020 09:28:55 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1911-01952.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### For EKiML,
```
@article{DBLP:journals/corr/abs-2010-08281,
  author    = {Wei Huang and
               Xingyu Zhao and
               Xiaowei Huang},
  title     = {Embedding and Synthesis of Knowledge in Tree Ensemble Classifiers},
  journal   = {CoRR},
  volume    = {abs/2010.08281},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.08281},
  archivePrefix = {arXiv},
  eprint    = {2010.08281},
  timestamp = {Wed, 21 Oct 2020 12:11:49 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-08281.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### For GUAP, 
```
@inproceedings{zhang2020generalizing,
      title={Generalizing Universal Adversarial Attacks Beyond Additive Perturbations}, 
      author={Yanghao Zhang and Wenjie Ruan and Fu Wang and Xiaowei Huang},
      year={2020},
      booktitle = {ICDM 2020}
}
```
