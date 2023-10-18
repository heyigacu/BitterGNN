## How to Use GNN
#### 1. install environment 
##### if use pip
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

