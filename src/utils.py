import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import torch
from typing import Optional, Tuple, Union
import json
import logging
import os
import datetime
import numpy as np
import random

def get_pos_index(predict,target):
    pos_index = []
    for u, u_list in enumerate(predict):
        u_result = []
        target[u] = [item for item in target[u] if item != 0]
        for i in u_list:
            if i != 0 and i in target[u]:
                u_result.append(1)
            else:
                u_result.append(0)

        u_result.append(len(target[u]))
        pos_index.append(u_result)
    return pos_index

def xavier_normal_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def xavier_uniform_initialization(module):
    """ initialize the parameters in nn.Embedding and nn.Linear layers. 
    For bias in nn.Linear layers,using constant 0 to initialize."""

    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError("Training loss is nan")

def get_logger(log_path = None,logname='log.out'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 定义控制台输出的handler
    formatter = logging.Formatter('%(asctime)s  --%(levelname)s --%(message)s')
    cmd_handler = logging.StreamHandler()
    cmd_handler.setLevel(logging.INFO)
    cmd_handler.setFormatter(formatter)

    logger.addHandler(cmd_handler)

    # 定义log文件的handler
    if log_path is not None:
        # 先检查路径并创建路径
        if not os.path.exists(log_path):
            os.makedirs(log_path) # 这个能创建多级目录

        log_file = os.path.join(log_path,logname)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def load_custom_attributes(obj, state_dict):
    for key in state_dict.keys():
        if hasattr(obj, key):
            setattr(obj, key, state_dict[key])
        else:
            print(f"Warning: attribute {key} does not exist in the model.")

def save_checkpoint(path,model=None):
    if model is not None:
        # 保存noise
        noise_model_file = path
        if isinstance(model,dict):
            torch.save(model, noise_model_file)
        else:
            torch.save(model.state_dict(), noise_model_file)
        print('{} is saved'.format(noise_model_file))


def load_checkpoint(path,model,device):
    state_dict = torch.load(path,map_location=device)
    print(state_dict.keys())
    if 'state_dict' in state_dict.keys():
        model.load_state_dict(state_dict.pop('state_dict'))
        load_custom_attributes(model,state_dict)# 加载额外的
    else: 
        model.load_state_dict(state_dict) 


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True