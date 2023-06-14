import yaml

from data_prepare import data_reader
from data_prepare.load_dict import load_random
from util.FileDumpLoad import dump_file, load_file

def load_data(config={}, reload=True):
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
    mid_cikm16_emb_dict = "mid_datacikm16_emb_dict.data"


    if reload:
        print( "reload the datasets.")
        print (config['dataset'])

        if config['dataset'] == 'rsc15_4':
            train_data, test_data, item2idx, n_items = data_reader.load_rsc15_data(
                rsc15_train,
                rsc15_test,
                pro = 4
            )

            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = '../datas/rsc15/embeddings/'
            dump_file([emb_dict, path+mid_rsc15_4_emb_dict])
            print("-----")

        if config['dataset'] == 'rsc15_64':
            train_data, test_data, item2idx, n_items = data_reader.load_rsc15_data(
                rsc15_train,
                rsc15_test,
                pro = 64
            )

            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx, edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = '../datas/rsc15/embeddings/'
            dump_file([emb_dict, path + mid_rsc15_64_emb_dict])
            print("-----")

        if config['dataset'] == 'cikm16':
            train_data, test_data, item2idx, n_items = data_reader.load_cikm16_data(
                cikm16_train,
                cikm16_test,
                class_num=config['class_num']
            )
            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = '../datas/cikm16/embeddings/'
            dump_file([emb_dict, path+mid_cikm16_emb_dict])
            print("-----")

    else:
        print ("not reload the datasets.")
        print(config['dataset'])

        if config['dataset'] == 'rsc15_4':
            train_data, test_data, item2idx, n_items = data_reader.load_rsc15_data(
                rsc15_train,
                rsc15_test,
                pro=4
            )

            config["n_items"] = n_items-1
            path = '../datas/rsc15/embeddings/'
            emb_dict = load_file(path + mid_rsc15_4_emb_dict)
            config['pre_embedding'] = emb_dict[0]
            # path = 'datas/mid_data'
            # dump_file([emb_dict, path+mid_rsc15_emb_dict])
            print("-----")

        if config['dataset'] == 'rsc15_64':
            train_data, test_data, item2idx, n_items = data_reader.load_rsc15_data(
                rsc15_train,
                rsc15_test,
                pro=64
            )

            config["n_items"] = n_items-1
            # emb_dict = load_random(n_items, edim=config['hidden_size'], init_std=config['emb_stddev'])
            # path = 'datas/train_emb/'
            # emb_dict = load_file(path + "rsc15_64_emb.data")
            path = '../datas/rsc15/embeddings/'
            emb_dict = load_file(path+mid_rsc15_64_emb_dict)
            config['pre_embedding'] = emb_dict[0]

            # dump_file([emb_dict, path + mid_rsc15_emb_dict])
            print("-----")

        if config['dataset'] == 'cikm16':
            train_data, test_data, item2idx, n_items = data_reader.load_cikm16_data(
                cikm16_train,
                cikm16_test,
                class_num=config['class_num']
            )
            config["n_items"] = n_items-1
            path = '../datas/rsc15/embeddings/'
            emb_dict = load_file(path + mid_cikm16_emb_dict)
            # path = 'datas/train_emb/'
            # emb_dict = load_file(path + "cikm16_emb.data")
            config['pre_embedding'] = emb_dict[0]
            print("-----")

    return train_data, test_data
