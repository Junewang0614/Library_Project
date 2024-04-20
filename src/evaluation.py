'''
一些指标运算
'''
import torch
import numpy as np

def compute_all_matrics(matrics,topk,topk_idx,pos_len_list,
                        prediction_list=None,
                        **kwargs):
    result_dict = {}
    for matric in matrics:
        if matric.lower() == 'ndcg':
            ans_dict = compute_ndcg(topk_idx,pos_len_list,topk)
            result_dict.update(ans_dict)
        elif matric.lower() == 'recall':
            ans_dict = compute_recall(topk_idx,pos_len_list,topk)
            result_dict.update(ans_dict)

        elif matric.lower() == 'itemcoverage':
            ans_dict = comput_itemcoverage(prediction_list,topk=topk,**kwargs) # 应该ok
            result_dict.update(ans_dict)
        else:
            print('The matric of {} has not been implemented yet.')

    return result_dict

def comput_itemcoverage(predict_list,tot_item_num,topk,**kwargs):
    # 假设predict_list已经是numpy的格式
    metric_dict = {}
    for k in topk:
        key = "{}@{}".format('itemcoverage',k)
        # get coverage计算
        now_list = predict_list[:,:k]
        unique_count = np.unique(now_list).shape[0] # 独特的个数
        value = unique_count / tot_item_num
        metric_dict[key] = value

    return metric_dict

def compute_ndcg(topk_idx, pos_len_list,topk):
    pos_index = topk_idx.to(torch.bool).numpy()
    pos_len = pos_len_list.squeeze(-1).numpy()

    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    avg_result = result.mean(axis=0) # 按照位置取结果就可以
    ans = {}
    for k in topk:
        key = "NDCG@{}".format(k)
        ans[key] = avg_result[k-1]

    return ans

def compute_recall(topk_idx, pos_len_list,topk):
    pos_index = topk_idx.to(torch.bool).numpy()
    pos_len = pos_len_list.squeeze(-1).numpy()

    result = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    avg_result = result.mean(axis=0)
    ans = {}
    for k in topk:
        key = "recall@{}".format(k)
        ans[key] = avg_result[k-1]

    return ans