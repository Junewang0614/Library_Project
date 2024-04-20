from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import pandas as pd
import os
from scipy.sparse import coo_matrix
import ipdb


def uni_sampling(item_num,start=1):
    return np.random.randint(start,item_num+start)

def padding(batch_data):
    padding_idx = 0
    max_length = max([len(data) for data in batch_data])
    batch_pad_data = []
    for data in batch_data:
        length_to_pad = max_length - len(data)
        data += [padding_idx] * length_to_pad
        batch_pad_data.append(data)
    
    return batch_pad_data

class InterTrainDataset(Dataset):
    def __init__(self,inters,user_seq,user_num,item_num) -> None:
        super().__init__()
        self.dataset = inters # (user，item)对组成的list
        self.user_seq = user_seq  # {uid:[intered list]}
        # todo:
        # self.inter_matrix
        # 这个是原始的数据，模型中需要每个都加1
        self.user_num = user_num
        self.item_num = item_num
        self.inter_matrix = self.get_inter_matrix()
    
    def get_inter_matrix(self):
        # 根据交互数据构造交互矩阵
        users = np.array([i for i,j in self.dataset])
        items = np.array([j for i,j in self.dataset])
        data = np.ones_like(users)

        inter_matrix = coo_matrix((data,(users,items)),shape=(self.user_num+1,
                                                              self.item_num+1))
        return inter_matrix



    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # return super().__getitem__(index)
        user_id,item_id = self.dataset[idx]
        intered_items = self.user_seq[user_id]
        # 采负样
        neg_id = uni_sampling(self.item_num)
        while neg_id in intered_items:
            neg_id = uni_sampling(self.item_num)
        
        return np.array(user_id),np.array(item_id),np.array(neg_id)

class InterEvalDataset(Dataset):
    def __init__(self,user_seq,user_his) -> None:
        super().__init__()
        self.user_seq = user_seq
        self.dataset = list(user_seq.keys())
        self.user_his = user_his
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user = self.dataset[idx]
        targets = self.user_seq[user]
        his_seqs = self.user_his.get(user,[])

        return user,targets,his_seqs
        # return np.array(user),np.array(targets)

    def collate_fn(self, batch_data):
        # ipdb.set_trace()
        # users,targets = batch_data
        users = [u for u,_,_ in batch_data]
        targets = [tar for _,tar,_ in batch_data]
        his_seqs = [his for _,_,his in batch_data]

        # user变tensor
        users = torch.LongTensor(users)
        # targets padding
        targets = padding(targets)
        # 交互过的items变tensor
        his_seqs = torch.LongTensor(padding(his_seqs))

        return users,targets,his_seqs

class InterEvalReverseDataset(Dataset):
    def __init__(self,item_seq,item_his) -> None:
        super().__init__()
        self.item_seq = item_seq
        self.dataset = list(item_seq.keys())
        self.item_his = item_his
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        targets = self.item_seq[item]
        # 这里有item可能没出现过
        his_seqs = self.item_his.get(item,[])

        return item,targets,his_seqs
        # return np.array(user),np.array(targets)

    def collate_fn(self, batch_data):
        # ipdb.set_trace()
        # users,targets = batch_data
        items = [i for i,_,_ in batch_data]
        targets = [tar for _,tar,_ in batch_data]
        his_seqs = [his for _,_,his in batch_data]

        # user变tensor
        items = torch.LongTensor(items)
        # targets padding
        targets = padding(targets)
        # 交互过的items变tensor
        his_seqs = torch.LongTensor(padding(his_seqs))

        return items,targets,his_seqs


# def get_user_interlist(file,return_list = False,time_order=True):
def get_user_seq(inter_data):
    user_list = inter_data['user_id'].unique().tolist()
    user_inter_list = []
    user_inter_dict = {}
    for user in user_list:
        inter_list = inter_data[inter_data['user_id'] == user].sort_values(['timestamp'])['item_id'].tolist()
        user_inter_dict[user] = inter_list
        user_inter_list.append((user,inter_list))

    return user_inter_dict,user_inter_list

def get_item_seq(inter_data):
    item_list = inter_data['item_id'].unique().tolist()
    item_inter_list = []
    item_inter_dict = {}
    for item in item_list:
        inter_list = inter_data[inter_data['item_id'] == item].sort_values(['timestamp'])['user_id'].tolist()
        item_inter_dict[item] = inter_list
        item_inter_list.append((item,inter_list))

    return item_inter_dict,item_inter_list

def get_train_loader(path,user_num,item_num,
                     batch_size=20,num_workers=4):
    
    file = os.path.join(path,'train.csv')
    data_df = pd.read_csv(file)
    user_inter_dict,_ = get_user_seq(data_df)
    inter_data = list(zip(data_df['user_id'],data_df['item_id']))

    dataset = InterTrainDataset(inter_data,user_inter_dict,
                                user_num,item_num)
    
    dataloader = DataLoader(dataset,batch_size,shuffle=True,
                            num_workers=num_workers)
    
    return dataset,dataloader

def get_all_user_eval_loader(path,batch_size=20,num_workers=4):
    file = os.path.join(path,'lib-unrepeat.inter')
    data_df = pd.read_csv(file,sep='\t')
    data_df.columns = ['user_id','item_id','timestamp']

    user_inter_dict,_ = get_user_seq(data_df)
    user_his_dict = {}
    dataset = InterEvalDataset(user_inter_dict,user_his_dict)

    dataloader = DataLoader(dataset,batch_size,shuffle=False,num_workers=num_workers,
                            collate_fn=dataset.collate_fn)

    return dataset,dataloader

def get_all_item_eval_loader(path,batch_size=20,num_workers=4):
    file = os.path.join(path,'lib-unrepeat.inter')
    data_df = pd.read_csv(file,sep='\t')
    data_df.columns = ['user_id','item_id','timestamp']

    item_inter_dict,_ = get_item_seq(data_df)
    item_his_dict = {}
    dataset = InterEvalReverseDataset(item_inter_dict,item_his_dict)
    dataloader = DataLoader(dataset,batch_size,shuffle=False,num_workers=num_workers,
                            collate_fn=dataset.collate_fn)

    return dataset,dataloader

def get_eval_loader(path,split='val',batch_size=20,num_workers=4):
    file = os.path.join(path,'{}.csv'.format(split))
    data_df = pd.read_csv(file)
    his_file = os.path.join(path,'train.csv')
    his_df = pd.read_csv(his_file)

    user_inter_dict,_ = get_user_seq(data_df)
    user_his_dict,_ = get_user_seq(his_df)
    dataset = InterEvalDataset(user_inter_dict,user_his_dict)

    dataloader = DataLoader(dataset,batch_size,shuffle=False,num_workers=num_workers,
                            collate_fn=dataset.collate_fn)

    return dataset,dataloader

def get_eval_reverse_loader(path,split='val',batch_size=20,num_workers=4):
    file = os.path.join(path,'{}.csv'.format(split))
    data_df = pd.read_csv(file)
    his_file = os.path.join(path,'train.csv')
    his_df = pd.read_csv(his_file)

    item_inter_dict,_ = get_item_seq(data_df)
    item_his_dict,_ = get_item_seq(his_df)
    dataset = InterEvalReverseDataset(item_inter_dict,item_his_dict)
    dataloader = DataLoader(dataset,batch_size,shuffle=False,num_workers=num_workers,
                            collate_fn=dataset.collate_fn)

    return dataset,dataloader
