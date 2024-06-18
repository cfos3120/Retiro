import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import UnitTransformer
from .full_dataset import get_query_grid, create_a_normalizer

class unsupervised_Cavity_dataset_for_GNOT():
    def __init__(self,  
                 L=1.0, 
                 key_range_min = 1,
                 key_range_max = 100,
                 key_range_interval = 1,
                 resolution = 128,
                 normalize_y = False, 
                 normalize_x = False,
                 normalize_f = False,
                 reference_data_set = None, 
                 vertex = False, 
                 boundaries = False):

        print('\n Creating Keys Only Dataset')

    # Normalizer settings:
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.normalize_f = normalize_f
        self.resolution = resolution
        self.vertex = vertex
        self.boundaries = boundaries

        # Input Functions (Lid Velocities)
        self.data_lid_v = torch.tensor(np.round(np.arange(key_range_min,key_range_max,key_range_interval),1))# * 0.1/0.01 #<- apply for Reynolds Number

        # Input Queries (Coordinates)
        self.queries = get_query_grid(L=L, nx=resolution, boundaries=boundaries, vertex=vertex)

        # Get Normalizers (from reference dataset, otherwise create new ones)
        if reference_data_set:
            print('Normalizers sourced from Reference Dataset')
            self.query_normalizer = reference_data_set.query_normalizer
            self.output_normalizer = reference_data_set.output_normalizer
            self.input_f_normalizer = reference_data_set.input_f_normalizer
        else:
            self.query_normalizer = create_a_normalizer(self.queries)
            self.input_f_normalizer = UnitTransformer(self.data_lid_v)

        # Normalize if required
        if self.normalize_x:
            self.queries = self.query_normalizer.transform(self.queries, inverse=False)
            print(f'    Queries Normalized with Means: {self.query_normalizer.mean} and Stds: {self.query_normalizer.std}')
            
        if self.normalize_f:
            self.data_lid_v = self.input_f_normalizer.transform(self.data_lid_v, inverse=False)
            print(f'    Keys Normalized with Means: {self.input_f_normalizer.mean} and Stds: {self.input_f_normalizer.std}')
        
        self.__update_dataset_config()

    def __update_dataset_config(self): 
        self.config = {
            'input_dim': self.queries.shape[-1],
            'theta_dim': 0,
            'output_dim': 3,
            'branch_sizes': [1]
        }

class unsupervised_CavityDataset(Dataset):
    def __init__(self,dataset, random_coords = True):
        print('\n   Creating Keys Only Dataloader')
        self.in_queries = dataset.queries
        self.in_keys_all = dataset.data_lid_v
        self.random_coords = random_coords

    def __len__(self):
        return len(self.in_keys_all)

    def __getitem__(self, idx):
        
        # randomize input coordinates
        if self.random_coords:
            indices = torch.randperm(self.in_queries.shape[0])
            reverse_indices = torch.argsort(indices)
        else:
            indices = torch.arange(self.in_queries.shape[0])
            reverse_indices = indices
        
        in_keys     = self.in_keys_all[idx].float().reshape(1,1)
        in_queries  = self.in_queries[indices,...].float()
        return in_queries, in_keys, reverse_indices
    

