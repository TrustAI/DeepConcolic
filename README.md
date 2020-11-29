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
usage: deepconcolic.py [-h] [--model MODEL] [--inputs DIR] --outputs DIR
                       [--criterion nc, ssc...] [--setup-only] [--init INT]
                       [--max-iterations INT] [--save-all-tests]
                       [--rng-seed SEED] [--labels FILE]
                       [--dataset {mnist,fashion_mnist,cifar10,OpenML:har}]
                       [--extra-tests DIR [DIR ...]] [--vgg16-model]
                       [--filters {LOF}] [--norm linf, l0] [--input-rows INT]
                       [--input-cols INT] [--input-channels INT]
                       [--cond-ratio FLOAT] [--top-classes INT]
                       [--layers LAYER [LAYER ...]] [--feature-index INT]
                       [--fuzzing] [--num-tests INT] [--num-processes INT]
                       [--sleep-time INT] [--dbnc-spec SPEC]

Concolic testing for neural networks

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         the input neural network model (.h5)
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
  --dataset {mnist,fashion_mnist,cifar10,OpenML:har}
                        selected dataset
  --extra-tests DIR [DIR ...]
                        additonal directories of test images
  --vgg16-model         vgg16 model
  --filters {LOF}       additional filters used to put aside generated test
                        inputs that are too far from training data (there is
                        only one filter to choose from for now; the plural is
                        used for future-proofing)
  --norm linf, l0       the norm metric
  --input-rows INT      input rows
  --input-cols INT      input cols
  --input-channels INT  input channels
  --cond-ratio FLOAT    the condition feature size parameter (0, 1]
  --top-classes INT     check the top-xx classifications
  --layers LAYER [LAYER ...]
                        test layers given by name or index
  --feature-index INT   to test a particular feature map
  --fuzzing             to start fuzzing
  --num-tests INT       number of tests to generate
  --num-processes INT   number of processes to use
  --sleep-time INT      fuzzing sleep time
  --dbnc-spec SPEC      Feature extraction and discretisation specification
```

The neural network model under tested is specified by ``--model`` and a set of raw test data should be given
by using ``--inputs``. Some popular datasets like MNIST and CIFAR10 can be directly specified by using the
``--dataset`` option directly. ``--criterion`` is used to choose the coverage
criterion and ``--norm`` helps select the norm metric to measure the distance between inputs. Some examples
to run DeepConcolic are in the following.

To run an MNIST model

```
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --dataset mnist --outputs outs/
```

To run an CIFAR10 model

```
python deepconcolic.py --model ../saved_models/cifar10_complicated.h5 --dataset cifar10 --outputs outs/
```

To test a particular layer
```
python deepconcolic.py --model ../saved_models/cifar10_complicated.h5 --dataset cifar10 --outputs outs/ --layers 2
```

To run MC/DC for DNNs on the CIFAR-10 model

```
python deepconcolic.py --model ../saved_models/cifar10_complicated.h5 --criterion ssc --cond-ratio 0.1 --dataset cifar10 --outputs outs
```

To run MC/DC for DNNs on the VGG16 model (with input images from the ``data`` sub-directory)

```
python deepconcolic.py --vgg16-model --inputs data/ --outputs outs --cond-ratio 0.1 --top-classes 5 --labels labels.txt --criterion ssc
```

To run Concolic Sign-sign-coverage (MC/DC) for DNNs on the MNIST model

```
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --dataset mnist --outputs outs --criterion ssclp
```

DeepConcolic nows supports an experimental fuzzing engine. Try ``--fuzzing`` to use it. The following command will result in: one ``mutants`` folder, one ``advs`` folder for adversarial examples and an adversarial list ``adv.list``.

```
python src/deepconcolic.py --fuzzing <br/> --model ./saved_models/mnist2.h5 --inputs data/mnist-seeds/ --outputs outs --input-rows 28 --input-cols 28
```

To run Concolic BN-based Feature coverage (BFCov) for DNNs on the MNIST model
```
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --criterion bfc --norm linf --dataset mnist --outputs outs --dbnc-spec ../dbnc/example.yaml
```
See [the example YAML specification](dbnc/example.yaml) for details on how to configure the BN-based abstraction.


To run Concolic BN-based Feature-dependence coverage (BFdCov) for DNNs on the MNIST model
```
python deepconcolic.py --model ../saved_models/mnist_complicated.h5 --criterion bfdc --norm linf --dataset mnist --outputs outs --dbnc-spec ../dbnc/example.yaml
```
You could adjust the following two parameters in the DBNC specification file defined by `--dbnc-spec` to dump the generated bayesian network to files `bn4trained.yml` and `bn4tests.yml`. 
 ```  
    dump_bn_with_trained_dataset_distribution: True,
    dump_bn_with_final_dataset_distribution: True,
 ```

### Concolic Testing on Lipschitz Constants for DNNs

To run Lipschitz Constant Testing, please refer to instructions in folder "Lipschitz Constant Testing".

# Dependencies
We suggest to create an environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html) as follows:
```
conda create --name deepconcolic
conda activate deepconcolic
conda install opencv 
pip3 install scikit-learn\>=0.22.1
pip3 install tensorflow\>=2.3
pip3 install pulp\>=2.3.1
pip3 install adversarial-robustness-toolbox\>=1.3
pip3 install pomegranate\>=0.13.4  
```

Note as of September 2020 one may need to append `--use-feature=2020-resolver` at the end of each `pip3 install` command-line to work-around errors in dependency resolution.  Further missing dependency errors for a package _p_ can then be solved by uninstalling/installing _p_. Note: if pomegranate installation failed with the above command, please try "pip3 install pomegranate==0.13.4" instead. The pip3 commands can be put al-together as the follows (works on Mac OS at Nov, 2020)

```
pip3 install scikit-learn tensorflow pulp adversarial-robustness-toolbox pomegranate==0.13.4 scipy numpy --use-feature=2020-resolver
```

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
