
import sys
import logging
import copy
from tracemalloc import start
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import socket
import numpy as np
from tensorboardX import SummaryWriter
import os.path as osp
import time
from datetime import timedelta

def record_data(target_list,train_writer,mark):
    for key_item in target_list:
        for j in range(len(target_list[key_item])):
            if j == 0:
                if 'compound' not in key_item and 'new' not in key_item:
                    train_writer.add_scalar('avg_{}/{}'.format(mark,key_item),target_list[key_item][j],j)
            else:
                train_writer.add_scalar('avg_{}/{}'.format(mark,key_item),target_list[key_item][j],j)



        

def train_wfold(args): 
    comment_name = '_{}_{}'.format(args['model_name'],args['dataset'])
    train_writer = SummaryWriter(comment=comment_name)
   
    device = copy.deepcopy(args['device'])

    cnn_list = {'top1': []} 
    nme_list = {'top1': []}
    for i_run in range(args['nb_runs']):
        folds_cnn_list = {'top1': []}  # fold数量
        folds_nme_list = {'top1': []}
        for i_fold in range(args['fold']):
            args['device'] = device
            cnn_curve, nme_curve, nme_mark =  _train(args,i_run,train_writer,i_fold)
            
            for key_item in folds_cnn_list:
                
                folds_cnn_list[key_item].append(cnn_curve[key_item])
                if nme_mark:
                    folds_nme_list[key_item].append(nme_curve[key_item])       

        for key_item in cnn_list:
            cnn_list[key_item].append(np.mean(folds_cnn_list[key_item],axis=0))
            nme_list[key_item].append(np.mean(folds_nme_list[key_item],axis=0))


    nb_run_cnn_list = {'top1': []}
    nb_run_nme_list = {'top1': []}
    nb_run_cnn_list_var = {'top1': []}
    nb_run_nme_list_var = {'top1': []}
  
    for key_item in cnn_list:
        nb_run_cnn_list[key_item] = np.mean(cnn_list[key_item],axis=0)
        nb_run_cnn_list_var[key_item] = np.std(cnn_list[key_item],axis=0)
        logging.info('{}: CNN:{}+-{}'.format(key_item,\
            nb_run_cnn_list[key_item], nb_run_cnn_list_var[key_item]))

        if nme_mark:
            nb_run_nme_list[key_item] = np.mean(nme_list[key_item],axis=0)
            nb_run_nme_list_var[key_item] = np.std(nme_list[key_item],axis=0)
            logging.info('{}: NME:{}+-{}'.format(key_item,\
                nb_run_nme_list[key_item],nb_run_nme_list_var[key_item]))

       
        total_num = np.shape(cnn_list[key_item])[1]
        
        
        nb_run_cnn_avg = cnn_list[key_item]
        if nme_mark:
            nb_run_nme_avg = nme_list[key_item]
       
        logging.info('Last: {}: CNN:{:.2f}+-{:.2f}'.format(key_item,\
            np.mean(nb_run_cnn_avg), np.std(nb_run_cnn_avg)))
       
        if nme_mark:
            logging.info('Last: {}: NME:{:.2f}+-{:.2f}'.format(key_item,\
                    np.mean(nb_run_nme_avg), np.std(nb_run_nme_avg)))
        
    record_data(nb_run_cnn_list,train_writer,'CNN')
    if nme_mark:
        record_data(nb_run_nme_list,train_writer,'NME')

def _train(args, i_run,train_writer,fold):
    
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    logfilename = 'logs/{}_{}_{}_{}_{}_{}_{}'.format(args['prefix'], args['seed'], args['model_name'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'],i_run,args,fold)
    model = factory.get_model(args['model_name'], args)
    nme_mark = False
    

    cnn_curve, nme_curve = {'top1': []}, {'top1': []}
    
    test_times = []
    for task in range(data_manager.nb_tasks):
        logging.info("[*] All params: {:.2f} M".format(count_parameters(model._network)/ 1000000.0))
        logging.info(
            "[*] Trainable params: {:.2f} M".format(count_parameters(model._network, True)/ 1000000.0)
        )
        
        model.incremental_train(data_manager)
        dict_name = {'task':task,'run':i_run}
        
        start = time.time()
        cnn_accy, nme_accy = model.eval_task(data_manager._increments,dict_name)
        end = time.time()
        test_times.append(end - start)
        logging.info('\n[*] Fini! '.ljust(64, '-'))
        logging.info(f'[!] Testing total time = {timedelta(seconds=end - start)}s')
        
        model.after_task()

        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        for key_item in cnn_curve:  
            cnn_curve[key_item].append(cnn_accy[key_item])
            logging.info('CNN {} curve: {}'.format(key_item,cnn_curve[key_item]))

        if nme_accy is not None:
            nme_mark  = True
            logging.info('NME: {}'.format(nme_accy['grouped']))
            for key_item in nme_curve:  # [top1, base, old, new, compound]
                nme_curve[key_item].append(nme_accy[key_item])
                logging.info('NME {} curve: {}'.format(key_item,nme_curve[key_item]))
    
    times = getattr(model, "times", None)   
    if times is not None:         
        logging.info('\n[*] Total Time! '.ljust(64, '*'))
        logging.info('Training Time:{} s'.format(timedelta(seconds=np.sum(model.times))))
        logging.info('Testing Time:{} s'.format(timedelta(seconds=np.sum(test_times))))
    return cnn_curve, nme_curve, nme_mark
 

def _set_device(args):
    device_type = args['device']
    
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus



def _set_random():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
