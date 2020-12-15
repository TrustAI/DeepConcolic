Lipophilicity is a dataset curated from ChEMBL database containing experimental results on octanol/water distribution coefficient (logD at pH=7.4). Due to the importance of lipophilicity in membrane permeability and solubility, the task is of high importance to drug development.

The data file contains a csv table, in which columns below are used:
     "smiles" - SMILES representation of the molecular structure
     "exp" - Measured octanol/water distribution coefficient (logD) of the compound, used as label

Model accuracy
loss: 0.0917 - rmse: 0.2371 - val_loss: 0.7052 - val_rmse: 0.6278

Reference:
Hersey, A. ChEMBL Deposited Data Set - AZ dataset; 2015. https://doi.org/10.6019/chembl3301361
