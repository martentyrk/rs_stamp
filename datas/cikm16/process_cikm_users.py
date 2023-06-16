# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
import math

import yaml

np.random.seed(42)

with open('paths.yaml', 'r') as file:
    paths = yaml.safe_load(file)

PATH_TO_ORIGINAL_DATA = paths['root_path']+paths['project_name']+'/datas/cikm16/raw/'
PATH_TO_PROCESSED_DATA = paths['root_path']+paths['project_name']+'/datas/cikm16/processed/'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train-item-views.csv', sep=';', header=0, usecols=[0,1,2,3,4], dtype={0:np.int32, 1:np.float64, 2:np.int64, 3:np.int32,4:str})

# data.columns = ['sessionId', 'TimeStr', 'itemId']
data['Time'] = data['eventdate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp()) #This is not UTC. It does not really matter.
del(data['eventdate'])

data = data[data['userId'].notna()] #only keep rows where users are not na.
data['userId'] = data['userId'].astype(np.int64)

# Keep sessions which are longer than one and which have had at least 5 items clicked.
session_lengths = data.groupby('sessionId').size()
data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
item_supports = data.groupby('itemId').size()
data = data[np.in1d(data.itemId, item_supports[item_supports>=5].index)]
session_lengths = data.groupby('sessionId').size()
data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]

print('Total data length:', len(data))

userIds = data['userId'].unique()
num_of_users = len(userIds)

train_split_amount = int(math.floor(num_of_users * 0.8))
test_split_amount = int(math.floor(num_of_users * 0.2))


# Choose random userIds for train and test.
session_train_ids = np.random.choice(userIds, size=train_split_amount, replace=False)
session_test_ids = [single_sample for single_sample in userIds if single_sample not in session_train_ids]

# Get the locations of these userIds in the DataFrame
session_train_loc = data[data['userId'].isin(session_train_ids)].index
session_test_loc = data[data['userId'].isin(session_test_ids)].index

#Use the locations to get the full rows in the dataframe.
train = data[data.index.isin(session_train_loc)].copy(deep=True)
test = data[data.index.isin(session_test_loc)].copy(deep=True)

#Remove userId column
train = train.drop(['userId'], axis=1)
test = test.drop(['userId'], axis=1)


print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.sessionId.nunique(), train.itemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'cikm16_train_user_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.sessionId.nunique(), test.itemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'cikm16_test_user.txt', sep='\t', index=False)
