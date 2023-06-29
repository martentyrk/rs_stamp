import pandas as pd
import time

import os

def save_results(config, recall, mrr, split='test'):
    suf = time.strftime("%Y%m%d%H%M", time.localtime())[4:]
    path = f"output/{split}_results.csv"
    results_dict = {'time':time.strftime("%Y%m%d%H%M", time.localtime()), 'model': config['model'], 'dataset': config['dataset'], '@k':config['cut_off'], 'kfolds':config['k_folds'], 'recall':recall, 'mrr':mrr}
    results = pd.DataFrame(results_dict, index=[0])
    if os.path.exists(path):
        results_df = pd.read_csv(path, index_col=0)
    else:
        results_df = pd.DataFrame()
    results_df = results_df.append(results, ignore_index=False)
    results_df.to_csv(path)
    return results_dict

def save_repeat_ratio_results(config, repeat_ratio):
    path = f"output/repeat_ratio_results.csv"
    results_dict = {'time':time.strftime("%Y%m%d%H%M", time.localtime()), 'model': config['model'], 'dataset': config['dataset'], '@k':config['cut_off'], 'kfolds':config['k_folds'], 'user_split':config['user_split'],'repeat_ratio':repeat_ratio}
    results = pd.DataFrame(results_dict, index=[0])
    if os.path.exists(path):
        results_df = pd.read_csv(path, index_col=0)
    else:
        results_df = pd.DataFrame()
    results_df = results_df.append(results, ignore_index=False)
    results_df.to_csv(path)
    return results_dict
