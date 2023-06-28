# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from random import sample
import datetime as dt
import math
import pickle

import yaml

np.random.seed(42)

with open('../../paths.yaml', 'r') as file:
    paths = yaml.safe_load(file)
    
    
#Paths to data folders
PATH_TO_ORIGINAL_DATA = paths['root_path']+paths['project_name']+'/datas/cikm16/raw/'
PATH_TO_PROCESSED_DATA = paths['root_path']+paths['project_name']+'/datas/cikm16/processed/'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train-item-views.csv', sep=';', header=0, usecols=[0,1,2,3,4], dtype={0:np.int32, 1:np.float64, 2:np.int64, 3:np.int32,4:str})

# data.columns = ['sessionId', 'userId', 'TimeStr', 'itemId']
data['Time'] = data['eventdate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp()) #This is not UTC. It does not really matter.
del(data['eventdate'])

data = data[data['userId'].notna()] #only keep rows where users are not na.
data['userId'] = data['userId'].astype(np.int64) #Correct the type for the users.
session_lengths = data.groupby('sessionId').size()
#Remove items where session len is shorter than 2
data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
item_supports = data.groupby('itemId').size()
# Remove items that occur less than 5 times
data = data[np.in1d(data.itemId, item_supports[item_supports>=5].index)]
session_lengths = data.groupby('sessionId').size()
data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]

print(len(data), 'data length')
userIds = data['userId'].unique()
num_of_users = len(userIds)

#Create the split sizes for train and test
train_split_amount = int(math.floor(num_of_users * 0.8))
test_split_amount = int(math.floor(num_of_users * 0.2))

#Choose the amount of ids randomly
session_train_ids = np.random.choice(userIds, size=train_split_amount, replace=False)
session_test_ids = [single_sample for single_sample in userIds if single_sample not in session_train_ids]

session_train_loc = data[data['userId'].isin(session_train_ids)].index
session_test_loc = data[data['userId'].isin(session_test_ids)].index

train = data[data.index.isin(session_train_loc)].copy(deep=True)
test = data[data.index.isin(session_test_loc)].copy(deep=True)

#Remove columns that will not be used for training
train = train.drop(['userId'], axis=1)
test = test.drop(['userId'], axis=1)

#Save the userID's so that we could use them for getting the exact same split for the NARM model
pickle.dump(session_train_ids, open(PATH_TO_PROCESSED_DATA + 'train_ids.txt', 'wb'))
pickle.dump(session_test_ids, open(PATH_TO_PROCESSED_DATA + 'test_ids.txt', 'wb'))

print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.sessionId.nunique(), train.itemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'cikm16_train_user_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.sessionId.nunique(), test.itemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'cikm16_test_user.txt', sep='\t', index=False)
