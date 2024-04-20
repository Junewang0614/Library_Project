import pandas as pd
import sys
sys.path.append('../')
sys.path.append("../src/")

from dataset import get_train_loader,get_eval_loader
from models import MF
from accelerate import Accelerator
import torch.optim as optim
from utils import *
import time
from tqdm import tqdm
from train_lightgcn import eval_process

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
    'save_path':"../models"
}
# eval config
eval_config = dict(
    topk=[5, 10,20,50],
    metrics=['recall', 'NDCG'],
    key_metric='NDCG@10'
)

def main():
    accelerator = Accelerator()
    data_path = train_config['data_path']
    set_seed(0)
    exper_path = os.path.join(train_config['save_path'],'mf_{}'.format(get_local_time()))
    logger = get_logger(exper_path)

     # 数据
    train_set,train_loader = get_train_loader(data_path,**lib_config,
                                              batch_size=train_config['batch_size'])
    valid_set,valid_loader = get_eval_loader(data_path,split='val',
                                             batch_size=train_config['batch_size'])
    test_set,test_loader = get_eval_loader(data_path,split='test',
                                           batch_size=train_config['batch_size'])

    # model
    model = MF(**lib_config,**mf_config)   
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])

    # training
    model,optimizer,train_loader,valid_loader = accelerator.prepare(
        model,optimizer,train_loader,valid_loader
    )

    eval_config.update({
        'item_num':lib_config['item_num']
    })

    logger.info(' ************************** The training config is: ***************************')
    # logger.info(vars(args))
    logger.info(train_config)
    logger.info(lib_config)
    logger.info(mf_config)
    logger.info(eval_config)
    logger.info(' ************************** The model structure is: ***************************')
    logger.info(model)

    best_result = None
    stop = 0
    best_epoch = 0
    best_path = os.path.join(exper_path,'model.pth')
    logger.info('the number of valid user is {}'.format(len(valid_set)))
    logger.info('the number of test user is {}'.format(len(test_set)))

    for epoch in range(1,train_config['epochs'] + 1):
        if stop >= train_config['early_stop']:
            logger.info(
                '==================Train early stop at epoch {},the best epoch is {} ==========================='.format(
                    epoch, best_epoch))
            break
        
        model.train()
        start_time = time.time()
        total_loss = 0
        iter_data = tqdm(train_loader) if train_config['show_process'] else train_loader  

        for batch in iter_data:
            optimizer.zero_grad()
            loss = model.calculate_loss(batch)
            loss = loss.mean()
            check_nan(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        spend_time = int((time.time() - start_time))
        logger.info('Epoch {}, Total Loss {}, Spend Time {}'.format(epoch, total_loss, spend_time))       

        # eval
        if epoch % train_config['eval_num'] == 0:
            eval_results = eval_process(model,valid_loader,eval_config)

            if best_result is None or eval_results[eval_config['key_metric']] > best_result[eval_config['key_metric']]: 
                # 保存模型
                save_checkpoint(best_path, model)
                # 更新指标
                best_result = eval_results.copy()
                best_epoch = epoch
                stop = 0
                # 记录指标
                logger.info(
                    'EPOCH {}, ======================== The best results ========================'.format(epoch))
                logger.info(best_result)
                logger.info('==' * 18)
            else:
                stop += train_config['eval_num']   

    # test
    load_checkpoint(best_path, model, accelerator.device)
    best_model, test_loader = accelerator.prepare(model, test_loader)
    test_results = eval_process(best_model,test_loader,eval_config)
    logger.info('================================= The test results ===================================')
    logger.info(test_results)



if __name__ == '__main__':
    main()