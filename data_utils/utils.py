#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import torch
import numpy as np
import torch
from argparse import ArgumentParser

# List of tensors object
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        return self.x[item]

# Get Seed  
def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({s})
    '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("=" * 50)
        print(message)
        print("=" * 50)


# Simple normalization layer
class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]
            
# Argument Parser
def parse_arguments():
    
    parser = ArgumentParser(description='GNOT (retiro) Artemis Training Study')
    parser.add_argument('--name'        , type=str  , default='test')
    parser.add_argument('--dir'         , type=str  , default='test_dir')
    parser.add_argument('--path'        , type=str  , default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
    parser.add_argument('--epochs'      , type=int  , default=1)
    parser.add_argument('--sub_x'       , type=float, default=8)
    parser.add_argument('--inference'   , type=int  , default=1)
    parser.add_argument('--layers'      , type=int  , default=3)
    parser.add_argument('--n_hidden'    , type=int  , default=32)
    parser.add_argument('--train_ratio' , type=float, default=0.7)
    parser.add_argument('--seed'        , type=int  , default=42)
    parser.add_argument('--lr'          , type=float, default=0.001)
    parser.add_argument('--batch_size'  , type=int  , default=4)
    parser.add_argument('--rand_cood'   , type=int  , default=1)
    parser.add_argument('--normalize_f' , type=int  , default=1)
    parser.add_argument('--DP'          , type=int  , default=0)
    parser.add_argument('--Optim'       , type=str  , default='Adamw')
    parser.add_argument('--Hybrid_type' , type=str  , default='Train')
    parser.add_argument('--scheduler'   , type=str  , default='Step')
    parser.add_argument('--step_size'   , type=int  , default=50)
    parser.add_argument('--init_w'      , type=int  , default=0)
    parser.add_argument('--datasplit'   , type=float, default=0.7)
    parser.add_argument('--ckpt_path'   , type=str  , default='None')
    parser.add_argument('--gating'      , type=int  , default=0)
    parser.add_argument('--Key_only_batches'      , type=int  , default=0)
    parser.add_argument('--Secondary_optimizer'   , type=int  , default=0)
    parser.add_argument('--dynamic_balance'   , type=int  , default=0)
    parser.add_argument('--ko_res'      , type=int  , default=32)
    parser.add_argument('--data_name'    , type=str  , default='test')

    return parser.parse_args()
