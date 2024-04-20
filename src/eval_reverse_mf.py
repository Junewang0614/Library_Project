'''
反向验证
'''
import pandas as pd
import sys
sys.path.append('../')
sys.path.append("../src/")

from dataset import get_eval_reverse_loader,get_train_loader
from models import MF
from accelerate import Accelerator
from utils import *
import time
from tqdm import tqdm
from evaluation import compute_all_matrics
from eval_reverse import eval_reverse_process
# import ipdb

# model_config
mf_config = {
    'embedding_size':512,
}
# dataset_config
lib_config = {
    'user_num':31849,
    'item_num':111479
}
# training config
train_config = {
    'epochs':200,
    'batch_size':1024,
    'learning_rate':0.001,
    'eval_num':1,
    'early_stop':30,
    'show_process':False,
    'data_path':"../datasets/",
    'save_path':"../models",
    'bert_embed':False
}
# eval config
eval_config = dict(
    topk=[5, 10,20,50],
    metrics=['recall', 'NDCG'],
    key_metric='NDCG@10'
)

def main():
    # todo: the trained model
    model_path = ""
    accelerator = Accelerator()
    data_path = train_config['data_path']
    device = accelerator.device
    set_seed(0)

    # 1. 加载数据
    # train_set,train_loader = get_train_loader(data_path,**lib_config,
    #                                           batch_size=train_config['batch_size'])
    
    valid_set,valid_loader = get_eval_reverse_loader(data_path,split='val',
                                                     batch_size=train_config['batch_size'])
    
    test_set,test_loader = get_eval_reverse_loader(data_path,split='test',
                                                     batch_size=train_config['batch_size'])

    print('the size of valid_set:',len(valid_set))
    print('the size of test_set:',len(test_set))
    # 加载模型
    model = MF(**lib_config,**mf_config)
    load_checkpoint(model_path, model, device)

    model,valid_loader,test_loader = accelerator.prepare(model,valid_loader,test_loader)

    # 验证
    eval_config.update({
        'user_num':lib_config['user_num']
    })
    valid_results = eval_reverse_process(model,valid_loader,eval_config,show_process=True)
    test_results = eval_reverse_process(model,test_loader,eval_config,show_process=True)
    print('the valid results:')
    print(valid_results)
    print('the test results:')
    print(test_results)


if __name__ == '__main__':
    main()


