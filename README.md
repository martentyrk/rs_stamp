# Exploring data preparation methods for better recommendations

Our paper aims to reproduce the results of the [STAMP paper](https://dl.acm.org/doi/abs/10.1145/3219819.3219950) ([GitHub](https://github.com/uestcnlp/STAMP)) and use the model to explore how different data preparation methods affect the model's performance. For comparability, we also implement the [NARM model](https://arxiv.org/abs/1711.04725) so that our results obtained with STAMP would be comparable to some baseline method. 

The aim of this README is to make the reproducibility of our results as easy as possible. Follow the steps given below.

# 0) Environments
We use two different environments in this project, one for running the STAMP and its related files and another one for NARM and its related files.

### STAMP environment
```
python==3.9
tensorflow==2.12
dill
numpy
pyyaml
```

### NARM environment
```
python==3.8
torch==1.5.0
torchvision==0.6.0
tqdm
pip
numpy
pillow
openssl
cudatoolkit
xz
zlib
```

# 1) Paths
In order for everything to work, update the paths in paths.yaml. The yoochoose path you may add after downloading the required file in step 2.


# 2) Data & preprocessing
Our paper leverages two datasets: Diginetica and Yoochoose

After downloading the two datasets, you can put them in the folder `datas\`, then process them by `process_rsc.py` and  `process_cikm.py` respectively.

### YOOCHOOSE: 
In order to download YOOCHOOSE, follow the instructions given here until step 2, we dont need step 3 since data is provided in the datas folder already, with .data files:
https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/YOOCHOOSE.md
From the downloaded files, we need to keep yoochoose-clicks.dat

To get the required data files run the following files in datas/rsc15:

## STAMP preprocessing

```
python -u datas/rsc15/process_rsc.py
``` 

## NARM preprocessing

```
python -u narm/datasets/preprocess.py /
       --dataset yoochoose
```


### DIGINETICA:

Check for the Data field to download everything here: https://competitions.codalab.org/competitions/11161#learn_the_details-data2

The file we need is train-item-views.csv

Once the file is downloaded and under datas/cikm16/raw
run the following two snippets

The first one divides the data by session and the second one by users. We also save the user id's of training and testing in order to have the same train/test split for NARM

## STAMP
```
python -u datas/cikm16/process_cikm.py
```
```
python -u datas/cikm16/process_cikm_users.py
```

## NARM
Files will appear under narm/datasets

```
python -u narm/datasets/preprocess.py --dataset diginetica
```

```
python -u narm/datasets/preprocess_user.py
```

# 3) Running the experiments

## Session based splits

### For diginetica:

#### STAMP
```
python -u cmain.py \
       -m stamp_cikm \
       -d cikm16 \
       --epoch 30 \
       --reload
```

#### NARM
```
python -u narm/main.py / 
       --train_path 'path to train_session.txt in narm/datasets/diginetica' / 
       --test_path 'path to test_session.txt in narm/datasets/diginetica' /
       --diginetica /
       --checkpoint 'checkpoint name'
```

### For Yoochoose

#### STAMP
For the 64 version:
```
python -u cmain.py \
       -m stamp_rsc \ 
       -d rsc15_64 \
       --epoch 30
```

For the 4 version:
```
python -u cmain.py \
       -m stamp_rsc \ 
       -d rsc15_4 \
       --epoch 30
```

#### NARM
For the 64 version:
```
python -u main.py \
       --train_path 'Path to narm/datasetsyoochoose1_64/train.txt' \
       --test_path 'Path to narm/datasetsyoochoose1_64/test.txt' \
       --checkpoint 'checkpoint name'
```

For the 4 version:
```
python -u main.py \
       --train_path 'Path to narm/datasetsyoochoose1_4/train.txt' \
       --test_path 'Path to narm/datasetsyoochoose1_4/test.txt' \
       --checkpoint 'checkpoint name'
```

## User-based split

We were only able to implement the user-based split on the Diginetica dataset, since yoochoose does not include user ID's.

#### STAMP
```
python -u cmain.py \
       -m stamp_cikm \
       -d cikm16 \
       --epoch 30 \
       --reload \
       --user_split
```

#### NARM
```
python -u narm/main.py \
       --train_path 'path to train_user.txt in narm/datasets/diginetica' \
       --test_path 'path to test_user.txt in narm/datasets/diginetica' \
       --checkpoint 'checkpoint name' \
       --diginetica \
       --user_split
```

## K-fold cross-validation

The K-fold cross-validation has been implemented for the STAMP model and the two datasets.

#### CIKM16 / Diginetica
The arguments below use an `@K` value of 10 (`--cutoff 10`), and 5 folds (`--kfolds 5`).
```
python cmain.py \
       -m stamp_cikm \
       -d cikm16 \
       --epoch 30 \
       --reload \
       --kfolds 5 \
       --cutoff 10
```

#### RSC15 / Yoochoose
Equivalently to the Diginetica parameters, with either `rsc15_4` or `rsc15_64` as the `-d` argument, and `stamp_rsc` as the `-m` argument.
```
python cmain.py \
       -m stamp_rsc \
       -d rsc15_4 \
       --epoch 30 \
       --reload \
       --kfolds 5 \
       --cutoff 10
```

