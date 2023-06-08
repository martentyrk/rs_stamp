import argparse
from logging import getLogger
from utils.utils import init_seed
from utils.logger import init_logger, set_color
from config.configurator import Config
from model.sequential_recommender.stamp import STAMP
from data.utils import get_dataloader, create_dataset
from data.cikm16data_read import load_data2
from data.rsyc15data_read_p import load_data_p
from data.load_dict import load_random
from data.FileDumpLoad import dump_file, load_file
from trainer import Trainer
from torch.utils.data import DataLoader

root_path = '/home/lcur2471'
project_name = '/rs_stamp'

rsc15_train = root_path + project_name +'/data_training/datarsc15_train_full.txt'
rsc15_test = root_path + project_name +'/data_training/datarsc15_test.txt'
mid_rsc15_train_data = "rsc15_train.data"
mid_rsc15_test_data = "rsc15_test.data"
mid_rsc15_emb_dict = "rsc15_emb_dict.data"
mid_rsc15_4_emb_dict = "rsc15_4_emb_dict.data"
mid_rsc15_64_emb_dict = "rsc15_64_emb_dict.data"


cikm16_train = root_path + project_name +'data_training/cikm16/cmki16_train_full.txt'
cikm16_test = root_path + project_name +'/cikm16/cmki16_test.txt'
mid_cikm16_emb_dict = "cikm16_emb_dict.data"

def load_tt_datas(config={}, reload=True):
    '''
    loda data.
    config: 获得需要加载的数据类型，放入pre_embedding.
    nload: 是否重新解析原始数据
    '''

    if reload:
        print( "reload the datasets.")
        print (config['dataset'])

        if config['dataset'] == 'rsc15_4':
            train_data, test_data, item2idx, n_items = load_data_p(
                rsc15_train,
                rsc15_test,
                pro = 4
            )

            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = 'datas/mid_data'
            dump_file([emb_dict, path+mid_rsc15_4_emb_dict])
            print("-----")

        if config['dataset'] == 'rsc15_64':
            train_data, test_data, item2idx, n_items = load_data_p(
                rsc15_train,
                rsc15_test,
                pro = 64
            )

            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx, edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = 'datas/mid_data'
            dump_file([emb_dict, path + mid_rsc15_64_emb_dict])
            print("-----")

        if config['dataset'] == 'cikm16':
            train_data, test_data, item2idx, n_items = load_data2(
                cikm16_train,
                cikm16_test,
                class_num=config['class_num']
            )
            config["n_items"] = n_items-1
            emb_dict = load_random(item2idx,edim=config['hidden_size'], init_std=config['emb_stddev'])
            config['pre_embedding'] = emb_dict
            path = 'datas/mid_data'
            dump_file([emb_dict, path+mid_cikm16_emb_dict])
            print("-----")

    else:
        print ("not reload the datasets.")
        print(config['dataset'])

        if config['dataset'] == 'rsc15_4':
            train_data, test_data, item2idx, n_items = load_data_p(
                rsc15_train,
                rsc15_test,
                pro=4
            )

            config["n_items"] = n_items-1
            path = 'datas/mid_data'
            emb_dict = load_file(path + mid_rsc15_4_emb_dict)
            config['pre_embedding'] = emb_dict[0]
            # path = 'datas/mid_data'
            # dump_file([emb_dict, path+mid_rsc15_emb_dict])
            print("-----")

        if config['dataset'] == 'rsc15_64':
            train_data, test_data, item2idx, n_items = load_data_p(
                rsc15_train,
                rsc15_test,
                pro=64
            )

            config["n_items"] = n_items-1
            # emb_dict = load_random(n_items, edim=config['hidden_size'], init_std=config['emb_stddev'])
            # path = 'datas/train_emb/'
            # emb_dict = load_file(path + "rsc15_64_emb.data")
            path = 'datas/mid_data'
            emb_dict = load_file(path+mid_rsc15_64_emb_dict)
            config['pre_embedding'] = emb_dict[0]

            # dump_file([emb_dict, path + mid_rsc15_emb_dict])
            print("-----")

        if config['dataset'] == 'cikm16':
            train_data, test_data, item2idx, n_items = load_data2(
                cikm16_train,
                cikm16_test,
                class_num=config['class_num']
            )
            config["n_items"] = n_items-1
            path = 'datas/mid_data'
            emb_dict = load_file(path + mid_cikm16_emb_dict)
            # path = 'datas/train_emb/'
            # emb_dict = load_file(path + "cikm16_emb.data")
            config['pre_embedding'] = emb_dict[0]
            print("-----")

    return train_data, test_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="diginetica-session",
        help="Benchmarks for session-based rec.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Whether evaluating on validation set (split from train set), otherwise on test set.",
    )
    parser.add_argument(
        "--valid_portion", type=float, default=0.1, help="ratio of validation set."
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()
    
    
    config_dict = {
        "USER_ID_FIELD": "session_id",
        "load_col": None,
        "neg_sampling": None,
        "benchmark_filename": ["train", "test"],
        "alias_of_item_id": ["item_id_list"],
        "topk": [20],
        "metrics": ["Recall", "MRR"],
        "valid_metric": "MRR@20",
    }
    
    config = Config(
        model="STAMP", dataset=f"{args.dataset}", config_dict=config_dict
    )
    
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)
    
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # train_dataset, test_dataset = dataset.build()
    # if args.validation:
    #     train_dataset.shuffle()
    #     new_train_dataset, new_test_dataset = train_dataset.split_by_ratio(
    #         [1 - args.valid_portion, args.valid_portion]
    #     )
    #     train_data = get_dataloader(config, "train")(
    #         config, new_train_dataset, None, shuffle=True
    #     )
    #     test_data = get_dataloader(config, "test")(
    #         config, new_test_dataset, None, shuffle=False
    #     )
    # else:
    #     train_data = get_dataloader(config, "train")(
    #         config, train_dataset, None, shuffle=True
    #     )
    #     test_data = get_dataloader(config, "test")(
    #         config, test_dataset, None, shuffle=False
    #     )
    
    train_data, test_data = load_tt_datas(config, False)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
    
    model = STAMP(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    # trainer loading and initialization
    trainer = Trainer(config, model)
    
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("test result", "yellow") + f": {test_result}")
    
    
    
    
    