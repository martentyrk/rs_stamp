# coding=utf-8
from optparse import OptionParser
import tensorflow.compat.v1 as tf
import pandas as pd

import numpy as np
import copy 
from tqdm import tqdm
import numpy as np
from data_prepare.entity.samplepack import Samplepack
from data_prepare import data_loader

from data_prepare.data_loader import load_data

from util.Config import read_conf
from util.Randomer import Randomer
from util.kfolds import split_k_folds


from util.save_results import save_results, save_repeat_ratio_results
import importlib

def load_conf(model, modelconf):
    '''
    model: 需要加载的模型
    modelconf: model config文件所在的路径
    '''
    # load model config
    model_conf = read_conf(model, modelconf)
    if model_conf is None:
        raise Exception("wrong model config path.", model_conf)
    module = model_conf['module']
    obj = model_conf['object']
    params = model_conf['params']
    params = params.split("/")
    paramconf = ""
    model = params[-1]
    for line in params[:-1]:
        paramconf += line + "/"
    paramconf = paramconf[:-1]
    # load super params.
    param_conf = read_conf(model, paramconf)
    return module, obj, param_conf


def option_parse():
    '''
    parse the option.
    '''
    parser = OptionParser()
    parser.add_option(
        "-m",
        "--model",
        action='store',
        type='string',
        dest="model",
        default='gru4rec'
    )
    parser.add_option(
        "-d",
        "--dataset",
        action='store',
        type='string',
        dest="dataset",
        default='rsc15'
    )
    parser.add_option(
        "-r",
        "--reload",
        action='store_true',
        dest="reload",
        default=False
    )
    parser.add_option(
        "-c",
        "--classnum",
        action='store',
        type='int',
        dest="classnum",
        default=3
    )

    parser.add_option(
        "-t",
        "--test_model",
        action='store_true',
        dest="test_model",
        default=True
    )
    parser.add_option(
        "-n",
        "--notsavemodel",
        action='store_true',
        dest="not_save_model",
        default=False
    )
    parser.add_option(
        "-p",
        "--modelpath",
        action='store',
        type='string',
        dest="model_path",
        default='output/saved_models/'
    )
    parser.add_option(
        "-i",
        "--inputdata",
        action='store',
        type='string',
        dest="input_data",
        default='test'
    )
    parser.add_option(
        "-e",
        "--epoch",
        action='store',
        type='int',
        dest="epoch",
        default=10
    )
    parser.add_option(
            "-k",
            "--cutoff",
            action='store',
            type='int',
            dest="cutoff",
            default=10
        )
    
    parser.add_option(
            "-f",
            "--kfolds",
            action='store',
            type='int',
            dest="kfolds",
            default=0
        )
    parser.add_option(
        "--user_split",
        action='store_true',
        help='use the user_split dataset'
    )
    (option, args) = parser.parse_args()
    return option


def main(options, modelconf="config/model.conf"):
    '''
    model: 需要加载的模型
    dataset: 需要加载的数据集
    reload: 是否需要重新加载数据，yes or no
    modelconf: model config文件所在的路径
    class_num: 分类的类别
    use_term: 是否是对aspect term 进行分类
    '''
    model = options.model
    dataset = options.dataset
    reload = options.reload
    class_num = options.classnum
    test_model = options.test_model
    is_save = not options.not_save_model
    user_split = options.user_split
    model_path = options.model_path

    input_data = options.input_data
    epoch = options.epoch

    module_name, obj, config = load_conf(model, modelconf)
    config['model'] = model
    config['user_split'] = user_split
    config['dataset'] = dataset
    config['class_num'] = class_num
    config['nepoch'] = epoch
    config['model_save_path'] = model_path
    config['test_model'] = test_model


    # metric @ k
    config['cut_off'] = options.cutoff
    config['k_folds'] = options.kfolds
    best_recall = 0
    best_model_graph = None
    train_data, test_data = load_data(config, reload, kfolds=options.kfolds)

    # setup randomer
    Randomer.set_stddev(config['stddev'])
    config_key = 'cikm_threshold_acc' if dataset=='cikm16' else 'recsys_threshold_acc'
    module = importlib.import_module(module_name)
    sent_data = test_data
    
    ### START K-FOLDS TRAINING
    if options.kfolds > 1:
        best_model_path = f"{config['model_save_path']}{config['model']}-{config['dataset']}-{config['k_folds']}folds-atk{config['cut_off']}-best_model.ckpt"

        for fold in tqdm(range(options.kfolds)):
            cur_fold_graph = tf.Graph()
            with cur_fold_graph.as_default():
                model = getattr(module, obj)(config)
                model.build_model()

                # need to do this after each fold
                if is_save or not test_model:
                    saver = tf.train.Saver(max_to_keep=30)
                else:
                    saver = None
                init = tf.global_variables_initializer()
                fold_train_data, fold_val_data = train_data[fold]
                with tf.Session() as sess:
                    sess.run(init)
                    val_results = model.train(sess, fold_train_data, fold_val_data, saver, threshold_acc=config[config_key])

                    # if the P@K is the best, save the model
                    if val_results['recall'] > best_recall:
                        best_recall = val_results['recall']
                        saver.save(sess, best_model_path)
                        best_model_graph = cur_fold_graph
                        best_model = model


    else:
        model_path = f"{config['model_save_path']}{config['model']}-{config['dataset']}-atk{config['cut_off']}-usersplit_{user_split}.ckpt"

        best_model_graph = tf.Graph()
        with best_model_graph.as_default():
            model = getattr(module, obj)(config)
            model.build_model()
            if is_save or not test_model:
                saver = tf.train.Saver(max_to_keep=30)
            else:
                saver = None
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                model.train(sess, train_data, test_data, saver, threshold_acc=config[config_key])
                saver.save(sess, model_path)
                if test_model:
                    saver.restore(sess, model_path)
                    recall, mrr, repeat_ratio = model.test(sess, sent_data)

    if test_model and options.kfolds > 1:
        with best_model_graph.as_default():
            with tf.Session() as sess:
                saver.restore(sess, best_model_path)
                recall, mrr, repeat_ratio = best_model.test(sess, sent_data)
    save_repeat_ratio_results(config, repeat_ratio)
    save_results(config, recall, mrr)
            

if __name__ == '__main__':
    options = option_parse()
    main(options)
