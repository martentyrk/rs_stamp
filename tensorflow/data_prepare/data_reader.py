import pandas as pd
import numpy as np
from data_prepare.entity.sample import Sample
from data_prepare.entity.samplepack import Samplepack

from util.kfolds import split_k_folds

def load_rsc15_data(train_file, test_file, kfolds, pro=None, pad_idx = 0):
    '''
    ret = [contexts, aspects, labels, positions] ,
    context.shape = [len(samples), None], None should be the len(context); 
    aspects.shape = [len(samples), None], None should be the len(aspect);
    labels.shape = [len(samples)]
    positions.shape = [len(samples), 2], the 2 means from and to.
    '''
    # the global param.
    items2idx = {}  # the ret
    items2idx['<pad>'] = pad_idx
    # load the data
    train_data = _load_rsc15_data(train_file, items2idx, kfolds, pro=pro, pad_idx=pad_idx)
    test_data = _load_rsc15_data(test_file, items2idx, kfolds, pad_idx=pad_idx, testset=True)
    item_num = len(items2idx.keys())
    return train_data, test_data, items2idx, item_num

def _load_rsc15_data(file_path, item2idx, kfolds, pro = None, pad_idx=0, testset=False):
    idx_cnt = 0
    data = pd.read_csv(file_path, sep='\t', dtype={'ItemId': np.int64})
    data.sort_values(['SessionId', 'Time'], inplace=True)  # 按照sessionid和时间升序排列
    session_data = list(data['SessionId'].values)
    item_event = list(data['ItemId'].values)
    if pro is not None:
        lenth = int(len(session_data) / pro)
        session_data = session_data[-lenth:]
        item_event = item_event[-lenth:]
        for i in range(len(session_data)):
            if session_data[i] != session_data[i+1]:
                break
        session_data = session_data[i + 1:]
        item_event = item_event[i + 1:]    
    samplepack = Samplepack()

    # List of Sample() objects
    samples = []

    now_id = 0
    sample = Sample()
    last_id = None
    click_items = []
    print('------------- IN FUNCT ----------')
    print(f'FOLDS = {kfolds}')
    for i, (s_id,item_id) in enumerate(zip(session_data, item_event)):
        #if i in [0, 1, 2, 3, 4]:
        #    print(i, '= click items: ', click_items)
        # first loop
        if last_id is None:
            last_id = s_id
        if s_id != last_id:
            item_dixes = []
            for item in click_items:
                if item not in item2idx:
                    if idx_cnt == pad_idx:
                        idx_cnt += 1
                    item2idx[item] = idx_cnt
                    idx_cnt += 1
                item_dixes.append(item2idx[item])
            in_dixes = item_dixes[:-1]
            out_dixes = item_dixes[1:]
            sample.id = now_id

            # last_id = session id
            sample.session_id = last_id

            sample.click_items = click_items
            sample.items_idxes = item_dixes
            sample.in_idxes = in_dixes
            sample.out_idxes = out_dixes
            samples.append(sample)
            sample = Sample()
            last_id =s_id
            click_items = []
            now_id += 1
        else:
            last_id = s_id
        click_items.append(item_id)
        #print(i, 'samples = ', samples)

    
    sample = Sample()
    item_dixes = []
    for item in click_items:
        # for the last clicks?
        if item not in item2idx:
            if idx_cnt == pad_idx:
                idx_cnt += 1
            item2idx[item] = idx_cnt
            idx_cnt += 1
        item_dixes.append(item2idx[item])
    in_dixes = item_dixes[:-1]
    out_dixes = item_dixes[1:]
    sample.id = now_id

    # last_id = session id
    sample.session_id = last_id
    
    sample.click_items = click_items
    sample.items_idxes = item_dixes
    sample.in_idxes = in_dixes
    sample.out_idxes = out_dixes
    samples.append(sample)

    # only interested in doing kfolds for the train set when kfolds > 0
    if (kfolds > 0) and (testset == False):
        samples = np.asarray(samples)

        # a list containing tuple pairs of (train_samplepack, val_samplepack)
        folded_samples = []
        folds = split_k_folds(samples, kfolds)
        for fold_train, fold_val in folds:
            train_samplepack = Samplepack()
            train_samplepack.samples = samples[fold_train]
            train_samplepack.init_id2sample()

            val_samplepack = Samplepack()
            val_samplepack.samples = samples[fold_val]
            val_samplepack.init_id2sample()

            folded_samples.append((train_samplepack, val_samplepack))

        return folded_samples

    samplepack.samples = samples
    samplepack.init_id2sample()


    return samplepack


def load_cikm16_data(train_file, test_file, kfolds, pad_idx=0, class_num = 3):
    '''
    ret = [contexts, aspects, labels, positions] ,
    context.shape = [len(samples), None], None should be the len(context); 
    aspects.shape = [len(samples), None], None should be the len(aspect);
    labels.shape = [len(samples)]
    positions.shape = [len(samples), 2], the 2 means from and to.
    '''
    # the global param.
    items2idx = {}  # the ret
    items2idx['<pad>'] = pad_idx
    idx_cnt = 0
    # load the data
    train_data, idx_cnt = _load_cikm16_data(train_file, items2idx, idx_cnt, pad_idx, class_num, kfolds)
    test_data, idx_cnt = _load_cikm16_data(test_file, items2idx, idx_cnt, pad_idx, class_num, kfolds)
    item_num = len(items2idx.keys())
    return train_data, test_data, items2idx, item_num



def _load_cikm16_data(file_path, item2idx, idx_cnt, pad_idx, class_num, kfolds):

    data = pd.read_csv(file_path, sep='\t', dtype={'itemId': np.int64})
    # return
    data.sort_values(['sessionId', 'Time'], inplace=True)  # 按照sessionid和时间升序排列
    samplepack = Samplepack()
    samples = []
    now_id = 0
    sample = Sample()
    last_id = None
    click_items = []


    for s_id,item_id in zip(list(data['sessionId'].values),list(data['itemId'].values)):
        if last_id is None:
            last_id = s_id
        if s_id != last_id:
            item_dixes = []
            for item in click_items:
                if item not in item2idx:
                    if idx_cnt == pad_idx:
                        idx_cnt += 1
                    item2idx[item] = idx_cnt
                    idx_cnt += 1
                item_dixes.append(item2idx[item])
            in_dixes = item_dixes[:-1]
            out_dixes = item_dixes[1:]
            sample.id = now_id
            sample.session_id = last_id
            sample.click_items = click_items
            sample.items_idxes = item_dixes
            sample.in_idxes = in_dixes
            sample.out_idxes = out_dixes
            samples.append(sample)
            sample = Sample()
            last_id =s_id
            click_items = []
            now_id += 1
        else:
            last_id = s_id
        click_items.append(item_id)
    sample = Sample()
    item_dixes = []
    for item in click_items:
        if item not in item2idx:
            if idx_cnt == pad_idx:
                idx_cnt += 1
            item2idx[item] = idx_cnt
            idx_cnt += 1
        item_dixes.append(item2idx[item])
    in_dixes = item_dixes[:-1]
    out_dixes = item_dixes[1:]
    sample.id = now_id
    sample.session_id = last_id
    sample.click_items = click_items
    sample.items_idxes = item_dixes
    sample.in_idxes = in_dixes
    sample.out_idxes = out_dixes
    samples.append(sample)
    samplepack.samples = samples
    samplepack.init_id2sample()
    return samplepack, idx_cnt


