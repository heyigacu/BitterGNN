## BitterGNN
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10018600.svg)](https://doi.org/10.5281/zenodo.10018600)
We created a new GNN (HGNN, combined four modules including edge attention, GAT, MPNN and Weave-GNN), and trained the predictors BitterGNN can distinguish bitter/non-bitter and bitter/sweet taste of molecule. 

## How to Use HGNN
#### 1. install environment 
##### if use pip (python >= 3.9)
``` 
pip install requirements.txt
```
##### or use docker
``` 
docker build -t my_image .
docker run -it my_image /bin/bash
```

#### 2. run BitterGNN predictor
```
python predictor.py -h
python predictor.py -t 0 -m 0 -i test.csv -o result.csv
```

#### 3. repeat the job of the paper
you can run compare_gnn.ipynb one by one to compare the performance of GNN

you can run compare_taste_predictors.ipynb one by one to compare the performance of Bitter predicotrs



## Acknowledgments
Supported by Graduate Innovation Fund of Jilin University (Project's number: 2022208)

