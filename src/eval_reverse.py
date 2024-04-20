'''
反向验证
'''
import pandas as pd
import sys
sys.path.append('../')
sys.path.append("../src/")

from dataset import get_eval_reverse_loader,get_train_loader
from models import LightGCN
from accelerate import Accelerator
from utils import *
import time
from tqdm import tqdm
from evaluation import compute_all_matrics
# import ipdb

# model_config
lightgcn_config = {
    'embedding_size':512,
    'n_layers':2
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
    # todo: trained model
    model_path = ""
    accelerator = Accelerator()
    data_path = train_config['data_path']
    device = accelerator.device
    set_seed(0)

    # 1. 加载数据
    train_set,train_loader = get_train_loader(data_path,**lib_config,
                                              batch_size=train_config['batch_size'])
    
    valid_set,valid_loader = get_eval_reverse_loader(data_path,split='val',
                                                     batch_size=train_config['batch_size'])
    
    test_set,test_loader = get_eval_reverse_loader(data_path,split='test',
                                                     batch_size=train_config['batch_size'])

    print('the size of valid_set:',len(valid_set))
    print('the size of test_set:',len(test_set))
    # 加载模型
    model = LightGCN(interaction_matrix=train_set.inter_matrix,**lib_config,
                     **lightgcn_config,
                device=accelerator.device)
    model_dict = torch.load(model_path,map_location=device)
    model.load_state_dict(model_dict['state_dict'])

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



@torch.no_grad()
def eval_reverse_process(model,eval_loader,eval_config,show_process=False):
    predict_list = []
    model.eval()
    device = eval_loader.device
    predict_matrix = None

    eval_loader = tqdm(eval_loader) if show_process else eval_loader

    with torch.no_grad():
        for batch in eval_loader:
            item,target,item_intered = batch
            batch_size = len(target)

            # todo: ground truth的处理

            # 获得scores
            scores = model.reverse_predict(item) #[batch_size,item_num]
            scores = scores.view(batch_size,-1)
            # print(scores.shape)
            # 获得predict,要遮盖的区域为1
            x_scattered = torch.zeros(scores.shape).to(device)
            x_scattered[:, 0] = 1.0
            # ipdb.set_trace()
            x_scattered = x_scattered.scatter(1,item_intered.data.long(),1.0)
            prediction = torch.where(x_scattered.bool(),-np.inf,scores)
            _, indices = torch.topk(prediction, max(eval_config['topk']))
            indices = indices.cpu().numpy().tolist()

            # 评估的矩阵
            # check
            pos_index = get_pos_index(indices,target)
            if predict_matrix is None:
                predict_matrix = torch.tensor(pos_index,dtype=torch.int32)
            else:
                pos_index = torch.tensor(pos_index,dtype=torch.int32)
                predict_matrix = torch.cat((predict_matrix,pos_index),dim=0)

            predict_list.extend(indices)
            
        predict_list = np.array(predict_list)
        topk_idx, pos_len_list = torch.split(predict_matrix,
                                            [max(eval_config['topk']), 1], dim=1)
        
        result_dict = compute_all_matrics(eval_config['metrics'], eval_config['topk'],
                                      topk_idx, pos_len_list,
                                      prediction_list=predict_list, tot_item_num=eval_config['user_num'])

        return result_dict


if __name__ == '__main__':
    main()


