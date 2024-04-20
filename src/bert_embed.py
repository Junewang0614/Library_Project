import json
import numpy as np
import ipdb
from torch.utils.data import Dataset,DataLoader
import torch 
from transformers import BertTokenizer, BertModel
from accelerate import Accelerator
from utils import set_seed
from tqdm import tqdm
from copy import deepcopy
import os

class ContentData(Dataset):
    def __init__(self,contents,tokenizer,
                max_length=512):
        super(ContentData, self).__init__()
        self.data = contents
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    def collate_fn(self, batch_data):
        token_data = self.tokenizer(batch_data,padding=True,
                                    max_length=self.max_length,truncation=True, 
                                    return_tensors="pt")
        
        return token_data
def get_item_content(file,
                     keys=['title','brand','categories'],
                     keep_keys=True) -> dict:
    assert '.json' in file,'The raw content file must be a json file'
    content_dict = {}
    with open(file,'r',encoding='utf-8') as f:
        raw_content_data = json.load(f)

    for item in raw_content_data:
        atom_id = int(item['item_id'])
        item_info = ''
        for key in keys:
            info = item.get(key, '')
            if info == '':
                continue
            if isinstance(info, list):  # categories的问题,只保留了第一个cate
                temp = []
                for subinfo in info:
                    temp += [cate for cate in subinfo[1:] if cate not in temp]
                info = ' '.join(temp)
            if not isinstance(info,str) and np.isnan(info): # NaN的问题
                continue
            if keep_keys:
                item_info = ' '.join([item_info, key, info])
            else:
                item_info = ' '.join([item_info, info])
            item_info = item_info.strip()

        content_dict[atom_id] = item_info

    return content_dict

if __name__ == '__main__':
    set_seed(0)
    info_file = "../datasets/item_info.json"
    bert_path = "bert-base-chinese" # pretrained bert
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)
    accelerator = Accelerator()

    cols = ['题目','作者','出版社','出版年份','语种']
    content_dict = get_item_content(info_file,keys=cols)
    # ipdb.set_trace()
    # print(content_dict[5])

    contents = ['padding'] + list(content_dict.values())
    dataset = ContentData(contents, bert_tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    
    bert_model,dataloader = accelerator.prepare(bert_model,dataloader)

    bert_model.eval()
    embedding_tensor=None
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # ipdb.set_trace()
            hidden_states = bert_model(**batch).last_hidden_state
            batch_embed = hidden_states[:,0] # 取[cls]
            if embedding_tensor is None:
                embedding_tensor = deepcopy(batch_embed)
            else:
                embedding_tensor = torch.cat((embedding_tensor,batch_embed),0)
    # ipdb.set_trace()
    print(embedding_tensor.shape)
    save_path = '../datasets/'
    bert_embed_file = os.path.join(save_path,'bert_embed.pth')
    embed_dict = {'bert':embedding_tensor}

    torch.save(embed_dict,bert_embed_file)