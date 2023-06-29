import yaml
import re

from data_prepare import data_reader
from data_prepare.load_dict import load_random
from util.FileDumpLoad import dump_file, load_file

def load_data(config={}, reload=True, kfolds=0):
    '''
    loda data.
    config: 获得需要加载的数据类型，放入pre_embedding.
    nload: 是否重新解析原始数据
    '''
    with open('../paths.yaml', 'r') as file:
        paths = yaml.safe_load(file)

    # the data path.
    root_path = paths['root_path']
    project_name = paths['project_name']

    # the pretreatment data path.
    rsc15_train = root_path + project_name +'/datas/rsc15/processed/rsc15_train_full.txt'
    rsc15_test = root_path + project_name +'/datas/rsc15/processed/rsc15_test.txt'
    mid_rsc15_train_data = "rsc15_train.data"
    mid_rsc15_test_data = "rsc15_test.data"
    mid_rsc15_emb_dict = "rsc15_emb_dict.data"
    mid_rsc15_4_emb_dict = "rsc15_4_emb_dict.data"
    mid_rsc15_64_emb_dict = "rsc15_64_emb_dict.data"


    cikm16_train = root_path + project_name +'/datas/cikm16/processed/cikm16_train_full.txt'
    cikm16_test = root_path + project_name +'/datas/cikm16/processed/cikm16_test.txt'
    cikm16_train_users = root_path + project_name +'/datas/cikm16/processed/cikm16_train_user_full.txt'
    cikm16_test_users = root_path + project_name +'/datas/cikm16/processed/cikm16_test_user.txt'
    mid_cikm16_emb_dict = "mid_datacikm16_emb_dict.data"

    pro = int(re.search(r"([0-9]+)$", config['dataset']).group(1)) # extract the 'pro' value from the dataset string (rsc15_val)

    if reload:
        print( "reloading the datasets.")
        if config['dataset'][:5] == 'rsc15':
            train_data, test_data, item2idx, n_items = data_reader.load_rsc15_data(
                rsc15_train,
                rsc15_test,
                kfolds,
                pro = pro
            )

            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = '../datas/rsc15/embeddings/'

            if pro == 4:
                dump_file([emb_dict, path+mid_rsc15_4_emb_dict])
            elif pro == 64:
                dump_file([emb_dict, path+mid_rsc15_64_emb_dict])

        if config['dataset'] == 'cikm16':
            if config['user_split']:
                train_data, test_data, item2idx, n_items = data_reader.load_cikm16_data(
                    cikm16_train_users,
                    cikm16_test_users,
                    kfolds,
                    class_num=config['class_num']
                )
            else:
                train_data, test_data, item2idx, n_items = data_reader.load_cikm16_data(
                    cikm16_train,
                    cikm16_test,
                    kfolds,
                    class_num=config['class_num']
                )
            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = '../datas/cikm16/embeddings/'
            dump_file([emb_dict, path+mid_cikm16_emb_dict])

    else:
        if config['dataset'][:5] == 'rsc15':
            train_data, test_data, item2idx, n_items = data_reader.load_rsc15_data(
                rsc15_train,
                rsc15_test,
                kfolds,
                pro=pro
            )

            config["n_items"] = n_items-1
            path = '../datas/rsc15/embeddings/'
            if pro == 4:
                emb_dict = load_file(path + mid_rsc15_4_emb_dict)
            elif pro == 64:
                emb_dict = load_file(path + mid_rsc15_64_emb_dict)
            config['pre_embedding'] = emb_dict[0]

        if config['dataset'] == 'cikm16':
            if config['user_split']:
                train_data, test_data, item2idx, n_items = data_reader.load_cikm16_data(
                    cikm16_train_users,
                    cikm16_test_users,
                    kfolds,
                    class_num=config['class_num']
                )
            else:
                train_data, test_data, item2idx, n_items = data_reader.load_cikm16_data(
                    cikm16_train,
                    cikm16_test,
                    kfolds,
                    class_num=config['class_num']
                )
            config["n_items"] = n_items-1
            path = '../datas/rsc15/embeddings/'
            emb_dict = load_file(path + mid_cikm16_emb_dict)
            config['pre_embedding'] = emb_dict[0]

    return train_data, test_data
