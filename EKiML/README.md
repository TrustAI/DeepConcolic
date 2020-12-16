# EKiML-embed-knowledge-into-ML-model
Embed and synthesise the knowledge into random forest 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install essential packages for EKiML.

```bash
pip install scipy scikit-learn pysmt
```

## Usage

--Dataset : 'iris', 'breast_cancer', 'mushroom', 'nursery, 'cod-rna', 'sensorless', 'mnist' (or you can add your own data into load_data.py).

--Mode : 'embedding', 'synthesis'

--Embedding_Method : 'black-box', 'white-box'

--Model : 'forest', 'tree'

--Pruning : True, False

--SaveModel : True, False

--output : './model/'

```python
python main.py --Dataset iris --Mode embedding --Embedding_Method black-box --Model forest
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

