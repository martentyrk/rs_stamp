# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt

import yaml

with open('paths.yaml', 'r') as file:
    paths = yaml.safe_load(file)

#Paths to data folders
PATH_TO_ORIGINAL_DATA = paths['root_path']+paths['project_name']+'/datas/cikm16/raw/'
PATH_TO_PROCESSED_DATA = paths['root_path']+paths['project_name']+'/datas/cikm16/processed/'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train-item-views.csv', sep=';', header=0, usecols=[0,2,3,4], dtype={0:np.int32, 1:np.int64, 2:np.int32,3:str})
# data.columns = ['sessionId', 'TimeStr', 'itemId']
data['Time'] = data['eventdate'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d').timestamp()) #This is not UTC. It does not really matter.
del(data['eventdate'])

session_lengths = data.groupby('sessionId').size()
#Remove items where session len is shorter than 2
data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
item_supports = data.groupby('itemId').size()
# Remove items that occur less than 5 times
data = data[np.in1d(data.itemId, item_supports[item_supports>=5].index)]
session_lengths = data.groupby('sessionId').size()
data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]

tmax = data.Time.max()
session_max_times = data.groupby('sessionId').Time.max()
session_train = session_max_times[session_max_times < tmax-86400*7].index
session_test = session_max_times[session_max_times > tmax-86400*7].index

train = data[np.in1d(data.sessionId, session_train)]
trlength = train.groupby('sessionId').size()
train = train[np.in1d(train.sessionId, trlength[trlength>=2].index)]
test = data[np.in1d(data.sessionId, session_test)]
test = test[np.in1d(test.itemId, train.itemId)]
tslength = test.groupby('sessionId').size()
test = test[np.in1d(test.sessionId, tslength[tslength>=2].index)]
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.sessionId.nunique(), train.itemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'cikm16_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.sessionId.nunique(), test.itemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'cikm16_test.txt', sep='\t', index=False)
