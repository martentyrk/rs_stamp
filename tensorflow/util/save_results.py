import pandas as pd
import time

import os

def save_results(config, recall, mrr, split='test'):
    suf = time.strftime("%Y%m%d%H%M", time.localtime())[4:]
    path = f"output/{split}_results.csv"
    
    results = {'time':time.strftime("%Y%m%d%H%M", time.localtime()), 'model': config['model'], 'dataset': config['dataset'], '@k':config['cut_off'], 'kfolds':config['k_folds'], 'recall':recall, 'mrr':mrr}
    if os.path.exists(path):
        results_df = pd.read_csv(path)
    else:
        results_df = pd.DataFrame()
    results_df = results_df.append(results, ignore_index=True)
    results_df.to_csv(path)
