#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 17 Sep, 2019

Reference: https://github.com/CRIPAC-DIG/SR-GNN/blob/master/datasets/preprocess.py

预处理基本流程：
1. 创建两个字典sess_clicks和sess_date来分别保存session的相关信息。两个字典都以sessionId为键，其中session_click以一个Session中用户先后点击的物品id
构成的List为值；session_date以一个Session中最后一次点击的时间作为值，后续用于训练集和测试集的划分；
2. 过滤长度为1的Session和出现次数小于5次的物品；
3. 依据日期划分训练集和测试集。其中Yoochoose数据集以最后一天时长内的Session作为测试集，Diginetica数据集以最后一周时长内的Session作为测试集；
4. 分解每个Session生成最终的数据格式。每个Session中以不包括最后一个物品的其他物品作为特征，以最后一个物品作为标签。同时把物品的id重新编码成从1开始递增的自然数序列
"""

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name')
args = parser.parse_args()


if args.dataset == 'diginetica':
    dataset = '/home/lcur2471/rs_stamp/datas/cikm16/raw/train-item-views.csv'
    train_ids = '/home/lcur2471/rs_stamp/datas/cikm16/processed/train_ids.txt'
    test_ids = '/home/lcur2471/rs_stamp/datas/cikm16/processed/test_ids.txt'
    
    with open(train_ids, 'rb') as f, open(test_ids, 'rb') as test_f:
        train_user_ids = pickle.load(f)
        test_user_ids = pickle.load(test_f)




print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    sess_user = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in tqdm(reader):
        sessid = data['sessionId']
        if data['userId'] == 'NA':
            continue
        
        userId = int(data['userId'])
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date, userId
        curid = sessid
        
        item = data['itemId'], int(data['timeframe'])
        curdate = data['eventdate']

        sess_user[sessid] = userId
        
        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
            
        ctr += 1
    
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    
    for i in list(sess_clicks):
        sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
        sess_clicks[i] = [c[0] for c in sorted_clicks]
        
    # sorted_users = sorted(sess_user[i], key=operator.itemgetter(1))
    sess_date[curid] = date, userId
print("-- Reading data @ %ss" % datetime.datetime.now())

# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_user[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid[0] in iid_counts:
            iid_counts[iid[0]] += 1
        else:
            iid_counts[iid[0]] = 1
            

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i[0]] >= 5, curseq))
    if len(filseq) < 2:
        del sess_user[s]
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

dates = list(sess_date.items())

# Give sess variables the form they would have without user_id, then we can reuse original code.
tra_sess = [(x[0], x[1][0]) for x in dates if x[1][1] in list(train_user_ids)]
tes_sess = [(x[0], x[1][0]) for x in dates if x[1][1] in list(test_user_ids)]


# # Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))
print(len(tra_sess))
print(len(tes_sess))
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print(item_ctr, 'here')     # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)

print('Len of train and test seqs respectively')
print(len(tr_seqs))
print(len(te_seqs))
all = 0

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))

if not os.path.exists('diginetica'):
    os.makedirs('diginetica')
pickle.dump(tra, open('diginetica/train_user.txt', 'wb'))
pickle.dump(tes, open('diginetica/test_user.txt', 'wb'))
pickle.dump(tra_seqs, open('diginetica/all_train_seq_user.txt', 'wb'))

print('Done.')
