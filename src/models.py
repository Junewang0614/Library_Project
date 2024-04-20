'''
模型文件
'''
# import sys
# sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from utils import xavier_uniform_initialization,xavier_normal_initialization

class MF(nn.Module):
    def __init__(self,user_num,item_num,
                 embedding_size,) -> None:
        super().__init__()

        self.latent_dim = embedding_size
        self.n_users = user_num + 1 
        self.n_items = item_num + 1 

        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.loss = BPRLoss()

        self.apply(xavier_normal_initialization)

    def forward(self,user,item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        return user_e,item_e
    
    def calculate_loss(self,batch_data):
        user,pos_item,neg_item = batch_data
        user_e,pos_e = self.forward(user,pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_item_score = torch.mul(user_e, pos_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_e).sum(dim=1)

        loss = self.loss(pos_item_score, neg_item_score)
        return loss
    
    def predict(self,users): # 返回打分
        batch_size = users.shape[0]
        u_embeddings = self.user_embedding(users)

        scores = torch.matmul(u_embeddings,self.item_embedding.weight.transpose(0,1))

        return scores.view(batch_size,-1)
    
    # 提供item 计算出user
    def reverse_predict(self,items):
        # 返回打分，[batch,user_nums]
        batch_size = items.shape[0]
        i_embeddings = self.item_embedding(items)
        scores = torch.matmul(i_embeddings,self.user_embedding.weight.transpose(0,1))

        return scores.view(batch_size,-1)


class LightGCN(nn.Module):
    def __init__(self,user_num,item_num,
                 embedding_size,n_layers,device,interaction_matrix=None,
                 reg_weight=1e-05,
                 ):
        super().__init__()
        # 属性
        self.interaction_matrix = interaction_matrix.astype(np.float32) # 交互图
        self.latent_dim = embedding_size
        self.n_layers = n_layers
        self.device = device
        self.reg_weight = reg_weight

        # 添加一维padding
        self.n_users = user_num + 1 
        self.n_items = item_num + 1 

        # 网络
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )


        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        self.restore_user_e = None
        self.restore_item_e = None

        # # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        # parameters initialization
        self.apply(xavier_uniform_initialization)


    # embed初始化
    def initialization_with_pre_embed(self,new_weight):
        print('before equal situation:')
        print(torch.equal(self.item_embedding.weight,new_weight))
        assert self.item_embedding.weight.shape == new_weight.shape
        self.item_embedding.weight.data.copy_(new_weight)
        print('after equal situation:')
        print(torch.equal(self.item_embedding.weight,new_weight))


    def forward(self):
        # 
        all_embeddings = self.get_ego_embeddings() # [user+item,dim]
        embeddings_list = [all_embeddings] # e_0

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings) # e_{layer_idx}
        # 堆叠
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1) #[user+item,layer,dim]
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1) # 对应层对应位置求平均

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings
    
    # 输入user,返回打分 [batch_size,item_num]
    def predict(self,users):
        batch_size = users.shape[0]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        u_embeddings = self.restore_user_e[users]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(batch_size,-1)

    def calculate_loss(self,batch_data):
        user,pos_item,neg_item = batch_data

        # 每次更新都要重置一下保存的内容
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # 这里获得的都是e_0
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            # require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss       

    def reverse_predict(self,items):
        batch_size = items.shape[0]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        
        i_embeddings = self.restore_item_e[items]

        # dot with all item embedding to accelerate
        scores = torch.matmul(i_embeddings, self.restore_user_e.transpose(0, 1))

        return scores.view(batch_size,-1)


    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def get_norm_adj_mat(self):
         # 交互矩阵归一化
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )

        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        
        return SparseL

    








# ============== loss 函数 =======================
class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss