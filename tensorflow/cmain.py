# coding=utf-8
from optparse import OptionParser
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np


from data_prepare.entity.samplepack import Samplepack
from data_prepare import data_loader

from data_prepare.data_loader import load_data

from util.Config import read_conf
from util.Randomer import Randomer
from util.kfolds import split_k_folds

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
        "-a",
        "--nottrain",
        action='store_true',
        dest="not_train",
        default=False
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
        #paths['root_path']+paths['project_name']+'/tensorflow
        default='/output/saved_models/'
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
    is_train = not options.not_train
    is_save = not options.not_save_model
    model_path = options.model_path#paths['root_path']+paths['project_name']+options.model_path+model+dataset+'.ckpt'

    input_data = options.input_data
    epoch = options.epoch

    module, obj, config = load_conf(model, modelconf)
    config['model'] = model
    config['dataset'] = dataset
    config['class_num'] = class_num
    config['nepoch'] = epoch
    config['model_save_path'] = model_path
    
    # metric @ k
    config['cut_off'] = options.cutoff
    print(config)
    train_data, test_data = load_data(config, reload, kfolds=options.kfolds)
    print('------train data------')
    print(train_data)
    #<data_prepare.entity.samplepack.Samplepack object at 0x7f1156473350>
    #testing_data = train_data + test_data
    print('------train data------')

    #print(module)
    # model.STAMP_cikm
    
    module = __import__(module, fromlist=True)

    #print(module)
    # <module 'model.STAMP_cikm' from '/home/andre/Documents/GitHub/rs_stamp/tensorflow/model/STAMP_cikm.py'> 

    # setup randomer
    Randomer.set_stddev(config['stddev'])
    with tf.Graph().as_default():
        # build model
        model = getattr(module, obj)(config)
        model.build_model()
        if is_save or not is_train:
            saver = tf.train.Saver(max_to_keep=30)
        else:
            saver = None
        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_train:
                print(dataset)
                
                config_key = 'cikm_threshold_acc' if dataset=='cikm16' else 'recsys_threshold_acc'
                ######## TODO: TRAIN AND TEST ARE FOLDS IF KFOLDS > 0 
                if options.kfolds > 0:
                    for fold in range(options.kfolds):
                        fold_train_data, fold_val_data = train_data[fold]
                        model.train(sess, fold_train_data, fold_val_data, saver, threshold_acc=config[config_key])
                        
                else:
                    model.train(sess, train_data, test_data, saver, threshold_acc=config[config_key])

            else:
                # input data is test by default - it determines what dataset is tested on
                if input_data == "test":
                    sent_data = test_data
                elif input_data == "train":
                    if options.kfolds > 0:
                        sent_data = [fold for fold in train_data]
                    else:
                        sent_data = train_data
                else:
                    sent_data = test_data
                saver.restore(sess, model_path)
                model.test(sess, sent_data)

    # run test after training is finished
    print('--------------- TRAINING FINISHED, NOW TESTING ---------------')
    if input_data == "test":
        sent_data = test_data
    elif input_data == "train":
        if options.kfolds > 0:
            sent_data = [fold for fold in train_data]
        else:
            sent_data = train_data
    else:
        sent_data = test_data
        saver.restore(sess, model_path)
        model.test(sess, sent_data)
    print('--------------- TESTING FINISHED ---------------')


if __name__ == '__main__':
    options = option_parse()
    main(options)
