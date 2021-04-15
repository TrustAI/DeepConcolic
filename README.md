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

```sh
conda create --name deepconcolic python==3.7
conda activate deepconcolic
```
This should be followed by installing software dependencies:
```sh
conda install opencv nltk matplotlib
conda install -c pytorch torchvision
pip3 install numpy==1.19.5 scipy==1.4.1 tensorflow\>=2.4 pomegranate==0.14 scikit-learn scikit-image pulp keract np_utils adversarial-robustness-toolbox parse tabulate pysmt saxpy keras menpo patool z3-solver pyvis
```
# Download Example Models
We use Fashion-MNIST dataset as the running example. The following are two pre-trained mmodels, one larger and one smaller.
```sh
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
usage: python3 -m deepconcolic.main [-h] --dataset
                                    {OpenML:har,cifar10,fashion_mnist,mnist}
                                    --model MODEL --outputs DIR --criterion
                                    {nc,ssc,ssclp,bfc,bfdc} --norm {l0,linf}
                                    [--setup-only] [--init INT]
                                    [--max-iterations INT] [--save-all-tests]
                                    [--rng-seed SEED]
                                    [--extra-tests DIR [DIR ...]]
                                    [--filters {LOF}] [--norm-factor FLOAT]
                                    [--lb-hard FLOAT] [--lb-noise FLOAT]
                                    [--mcdc-cond-ratio FLOAT]
                                    [--top-classes CLS]
                                    [--layers LAYER [LAYER ...]]
                                    [--feature-index INT] [--dbnc-spec SPEC]
                                    [--dbnc-abstr PKL]

Concolic testing for Neural Networks

optional arguments:
  -h, --help            show this help message and exit
  --dataset {OpenML:har,cifar10,fashion_mnist,mnist}
                        selected dataset
  --model MODEL         the input neural network model (.h5 file or "vgg16")
  --outputs DIR         the output test data directory
  --criterion {nc,ssc,ssclp,bfc,bfdc}
                        the test criterion
  --norm {l0,linf}      the norm metric
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
  --extra-tests DIR [DIR ...], +i DIR [DIR ...]
                        additonal directories of test images
  --filters {LOF}       additional filters used to put aside generated test
                        inputs that are too far from training data (there is
                        only one filter to choose from for now; the plural is
                        used for future-proofing)
  --norm-factor FLOAT   norm distance upper threshold above which generated
                        inputs are rejected by the oracle (default is 1/4)
  --lb-hard FLOAT       hard lower bound for the distance between original and
                        generated inputs (concolic engine only---default is
                        1/255 for image datasets, 1/100 otherwise)
  --lb-noise FLOAT      extra noise on the lower bound for the distance
                        between original and generated inputs (concolic engine
                        only---default is 1/10)
  --mcdc-cond-ratio FLOAT
                        the condition feature size parameter (0, 1]
  --top-classes CLS     check the top-CLS classifications for models that
                        output estimations for each class (e.g. VGG*)
  --layers LAYER [LAYER ...]
                        test layers given by name or index
  --feature-index INT   to test a particular feature map
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

```sh
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --dataset mnist --outputs outs/
```

To run an CIFAR10 model

```sh
python -m deepconcolic.main --model saved_models/cifar10_complicated.h5 --dataset cifar10 --outputs outs/
```

To test a particular layer
```sh
python -m deepconcolic.main --model saved_models/cifar10_complicated.h5 --dataset cifar10 --outputs outs/ --layers 2
```

To run MC/DC for DNNs on the CIFAR-10 model

```sh
python -m deepconcolic.main --model saved_models/cifar10_complicated.h5 --criterion ssc --mcdc-cond-ratio 0.1 --dataset cifar10 --outputs outs
```

<!--  NB: temporary comment as --inputs argument is about to disapear:
To run MC/DC for DNNs on the VGG16 model (with input images from the ``data`` sub-directory)

```sh
python -m deepconcolic.main --model vgg16 --inputs data/ --outputs outs --mcdc-cond-ratio 0.1 --top-classes 5 --labels labels.txt --criterion ssc
```
-->

To run Concolic Sign-sign-coverage (MC/DC) for DNNs on the MNIST model

```sh
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --dataset mnist --outputs outs --criterion ssclp
```

## Bayesian Network based Abstraction

To run Concolic BN-based Feature coverage (BFCov) for DNNs on the MNIST model
```sh
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --criterion bfc --norm linf --dataset mnist --outputs outs --dbnc-spec dbnc/example.yaml
```
See [the example YAML specification](dbnc/example.yaml) for details on how to configure the BN-based abstraction.


To run Concolic BN-based Feature-dependence coverage (BFdCov) for DNNs on the MNIST model
```sh
python -m deepconcolic.main --model saved_models/mnist_complicated.h5 --criterion bfdc --norm linf --dataset mnist --outputs outs --dbnc-spec dbnc/example.yaml
```
You could adjust the following two parameters in the DBNC specification file defined by `--dbnc-spec` to dump the generated bayesian network to files `bn4trained.yml` and `bn4tests.yml`.
 ```yaml
    dump_bn_with_trained_dataset_distribution: True,
    dump_bn_with_final_dataset_distribution: True,
 ```


## Fuzzing Engine

DeepConcolic additionally features an experimental fuzzing engine.  The following command illustrates how to exercise this engine on a classifier for the CIFAR10 dataset: it will generate at most 1000 images obtained by mutating inputs randomly drawn from the CIFAR10 validation dataset, and save them into the ``outs/cifar10-fuzzing-basic`` directory.  Aversarial examples can be identified in the latter directory by searching for files named ``<test-number>-adv-<wrong-label>.png``, derived from file ``<test-number>-original-<true-label>.png``.  Passed tests are named in a similar way, as ``<test-number>-ok-<label>.png``.

```sh
python3 -m deepconcolic.fuzzer --dataset cifar10 --model saved_models/cifar10_complicated.h5 --processes 2 --outputs outs/cifar10-fuzzing-basic -N 1000
```

Further options are available to use this engine.  It is for instance possible to specify a set of files used as seeds for fuzzing with the option ``--inputs``, as in:

```sh
python3 -m deepconcolic.fuzzer --dataset mnist --model saved_models/mnist_complicated.h5  --inputs data/mnist-seeds --processes 5 --outputs outs/mnist-fuzzing-given-seeds -N 1000
```

or sample ``N`` inputs from the validation dataset beforehand with ``--sample N``:

```sh
python3 -m deepconcolic.fuzzer --dataset cifar10 --model saved_models/cifar10_complicated.h5 --sample 10 --processes 5 --outputs outs/cifar10-fuzzing-sub-sample10 -N 1000
```


## Working with Your Own Datasets

DeepConcolic provides means for working with additional datasets, that can be provided via a dedicate plugin system.
Such plugins are Python modules that are loaded when the tool starts, and are searched within any directory listed in the colon-separated environment variable `DC_PLUGINS_PATH` if this variable is defined, or else within the `./dc_plugins` directory if it exists (note the latter is relative to the current working directory).

Then, a new dataset can be registered by calling the `deepconcolic.datasets.register_dataset` function with a name for the dataset as first argument, and a function that loads and returns a dataset description as second argument.
The latter function must accept any set of named arguments (for future extensions), and return a tuple with: (i) a pair of arrays containting trainting data and labets; (ii) a similar pair for validation; (iii) the shape of each individual input element; (iv) a descriptor string in {`image`, `normalized`, `unknown`} (used for determining the input feature encoding—note the format of this descriptor is likely to be refined in future versions); and (v) a list of strings showing the individual label names.
The dataset arrays can be given using [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) or [`pandas.Dataframe`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) dataframes.
The typical pattern is as follows (for loading, e.g., the [MNIST dataset provided by `tensorflow`](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data), as already done in [`deepconcolic.datasets`](deepconcolic/datasets.py)):
```python
def load_mnist_data (**_):
  import tensorflow as tf
  img_shape = 28, 28, 1
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data ()
  x_train = x_train.reshape (x_train.shape[0], *img_shape).astype ('float32') / 255
  x_test = x_test.reshape (x_test.shape[0], *img_shape).astype ('float32') / 255
  return (x_train, y_train), (x_test, y_test), img_shape, 'image', \
         [ str (i) for i in range (0, 10) ]
register_dataset ('mnist', load_mnist_data)
```

For further illustrative purposes, we provide [an example dataset plugin](dc_plugins/toy_datasets/random.py), which can be used to randomly generate classification tasks.
This plugin registers several datasets (named, e.g., `rand10_2`, `rand10_5`, and `rand100_5`) upon startup of DeepConcolic, which should then show as valid options for the `--dataset` option.
We also provide a utility script to construct and train small DNNs for the above toy datasets:
To train a classifier for the `rand10_2` dataset, and then print a short classification report:
```sh
# The following saves the trained model under `/tmp' on Unix-style systems:
python3 -m utils.train4random rand10_2
python3 -m deepconcolic.eval_classifier --dataset rand10_2 --model /tmp/rand10_2_dense_50_50_10_10.h5
```
To run the fuzzer on the newly trained model, using a sample of 10 initial test data and 5 processes:
```sh
python3 -m deepconcolic.fuzzer --dataset rand10_2 --model /tmp/rand10_2_dense_50_50_10_10.h5 --sample 10 --processes 5 --outputs outs/rand10_2-fuzz1000 -N 1000
```
The above command outputs new inputs within a file `outs/rand10_2-fuzz1000/new_inputs.csv`.


# Tool 2 -- testRNN: Coverage Guided Testing for Recurrent Neural Networks

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

```sh
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
where:

1. `<modelName>` is in {`sentiment`, `mnist`, `fashion_mnist`, `ucf101`}
2. `<Num. of Test Cases>` is the expected number of test cases
3. `<Mutation Method>` is in {`random`, `genetic`}
4. `<SC threshold>` is in [0, 1]
5. `<BC threshold>` is in [0, 1]
6. `<Num. of symbols>` is in {1, 2, 3...}
7. `<seq in cells to test>` is in {`mnist: [4, 24], fashion_mnist: [4, 24], sentiment: [400, 499], ucf101: [0, 10]`}
8. `<modeName>` is in {`train`, `test`} with default value `test`
9. `<output directory>` specifies the path of the directory to save the output record and generated examples

For example, we can run the following
```sh
python -m testRNN.main --model fashion_mnist --TestCaseNum 10000 --Mutation random --threshold_SC 0.6 --threshold_BC 0.7 --symbols_TC 3 --seq [4,24] --outputs testRNN_output
```
which says that, we are working with Fashion-MNIST model, and the genetic algorithm based test case generation will terminate when the number of test cases is over 10000. We need to specify other parameters including threshold_SC, threshold_BC, symbols_TC, and seq. Moreover, the log is generated to the file testRNN_output/record.txt. Also the output of adversarial examples can be found in testRNN_output/adv_output

# Tool 3 -- EKiML: Embedding Knolwedge into Tree Ensembles

In this tool, we consider embedding knowledge into machine learning models. The knowledge expression we considered can express e.g., robustness and resilience to backdoor attack, etc. That is, we can "embed" knowledge into a tree ensemble, representing a backdoor attack on the tree ensemble. Also, we can "detect" if a tree ensemble has been attacked.

The paper is available at https://arxiv.org/pdf/2010.08281.pdf.

## Download pre-trained models

As the running example, we download the pre-trained HAR tree model as follows.

```sh
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

1. `<DatasetName>` is in {'iris', 'har', 'breast_cancer', 'mushroom', 'nursery, 'cod-rna', 'sensorless', 'mnist'}.
2. `<modeName>` is in {'embedding', 'synthesis'}, where 'synthesis' denotes the "extraction".
3. `<embeddingMethod>` is in {'black-box', 'white-box'}
4. `<modeType>` is in {'forest', 'tree'}
5. `<pruningFlag>` is in {True, False}, with default value False
6. `<saveModelFlag>` is in {True, False}, with default value False
7. `<workDirectory>` is the working directory, with default value 'EKiML_workdir'
8. `<Datadir>` is the directory where dataset files are located (default is 'EKiML/dataset')

For example, we can run the following
```sh
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

```sh
pip install torch torchvision matplotlib
```

## Download target Models
```sh
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
```sh
python run_fashion_mnist.py --cuda --gpuid 0 --resume
```
## Generalizing UAP for Cifar10:
```sh
python run_cifar.py --cuda --gpuid 0 --model VGG19 --tau 0.1 --eps 0.03
```
## Generalizing UAP for ImageNet:
```sh
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
