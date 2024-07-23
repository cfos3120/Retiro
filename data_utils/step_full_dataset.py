import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import UnitTransformer, MultipleTensors

def create_a_normalizer(un_normalized_data):

        # Flatten
        n_channels = un_normalized_data.shape[-1]
        batches_and_nodes = torch.prod(torch.tensor(un_normalized_data.shape[:-1])).item()
        all_features = un_normalized_data.reshape(batches_and_nodes,n_channels)
        
        return UnitTransformer(all_features)

class Step_2D_dataset_for_GNOT():
    def __init__(self, 
                data_path,
                normalize_y = False, 
                normalize_x = False,
                normalize_f = False,
                boundary_input_f = False,
                subsample_ratio = 1.0):
    
        print('\nCalculating Step Dataset Normalizers')
        # Normalizer settings:
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.normalize_f = normalize_f

        self.data_dict   = np.load(data_path,allow_pickle=True).item()
        self.Re = torch.tensor(self.data_dict['Re'])
        self.x = torch.tensor(self.data_dict['Points'])
        self.y = torch.tensor(self.data_dict['Solutions'])
        self.n_internal = self.data_dict['Points'].shape[0]
        self.n_cases = self.data_dict['Solutions'].shape[0]
        self.bc = None

        print('Dataset loaded has the following keys: ', self.data_dict.keys())
        print(f'Reynolds Number Range: {self.Re.min()}-{self.Re.max()}')
        print(f'Total Cases: {self.n_cases}')
        print(f'Total Internal Cells: {self.n_internal} (before any subsampling)')
        print(f'Patch Names: {self.data_dict["Boundary"].keys()}')

        # Store the indices of all patches
        self.patch_indices = dict.fromkeys(self.data_dict["Boundary"])
        starting_bc_index = self.x.shape[0]

        for patch in self.data_dict["Boundary"]:
            num_points = self.data_dict["Boundary Points"][patch].shape[0]
            self.x = torch.concat([self.x, torch.tensor(self.data_dict["Boundary Points"][patch])], dim=0)
            self.y = torch.concat([self.y, torch.tensor(self.data_dict["Boundary"][patch])], dim=1)
            self.patch_indices[patch] = np.arange(starting_bc_index,starting_bc_index+num_points)
            starting_bc_index += num_points

        if boundary_input_f: 
            self.bc = {'Coords' : self.x[self.n_internal:,...],
                       'Values' : self.y[:,self.n_internal:,...]
                       }

        print(f'Total Boundary Cells: {self.x.shape[0]-self.data_dict["Points"].shape[0]}\n')

        # Get Normalizers
        self.x_normalizer = create_a_normalizer(self.x)
        self.y_normalizer = create_a_normalizer(self.y)
        self.xi_normalizer = UnitTransformer(self.Re)

        # Normalize if required
        if self.normalize_x:
            self.queries = self.x_normalizer.transform(self.x, inverse=False)
            print(f'    Queries Normalized with Means: {self.x_normalizer.mean} and Stds: {self.x_normalizer.std}')
            if boundary_input_f:
                self.bc['Coords'] = self.x_normalizer.transform(self.bc['Coords'], inverse=False)
                print(f'    BC coords Keys Normalized with Means: {self.x_normalizer.mean} and Stds: {self.x_normalizer.std}')

        if self.normalize_y:
            self.data_out = self.y_normalizer.transform(self.y, inverse=False)
            print(f'    Solutions Normalized with Means: {self.y_normalizer.mean} and Stds: {self.y_normalizer.std}')
            if boundary_input_f:
                self.bc['Values'] = self.y_normalizer.transform(self.bc['Values'], inverse=False)
                print(f'    BC values Keys Normalized with Means: {self.y_normalizer.mean} and Stds: {self.y_normalizer.std}')
    
        if self.normalize_f:
            self.Re = self.xi_normalizer.transform(self.Re, inverse=False)
            print(f'    Re Keys Normalized with Means: {self.xi_normalizer.mean} and Stds: {self.xi_normalizer.std}')
        
        # Only Subsample the internal field (and after normalizer are calculated)
        if subsample_ratio != 1.0:
            # Interior random samples
            sample = list(np.random.default_rng().choice(self.n_internal, size=int(self.n_internal*subsample_ratio), replace=False))
            # All boundary points
            sample += list(np.arange(self.n_internal,self.x.shape[0]))
            # points reduced
            n_points_reduced = self.n_internal - int(self.n_internal*subsample_ratio)

            self.x = self.x[sample,...]
            self.y = self.y[:,sample,...]

            for patch in self.patch_indices:
                self.patch_indices[patch] = self.patch_indices[patch] - n_points_reduced

            print(f'\nTotal Internal Cells reduced by {n_points_reduced} to: {self.x.shape[0]} (after subsampling)')

        self.__update_dataset_config()

    def __update_dataset_config(self): 
        self.config = {
            'input_dim': self.x.shape[-1],
            'theta_dim': 0,
            'output_dim': self.y.shape[-1],
            'branch_sizes': [3 if self.bc is not None else 1]
        }

class StepDataset():
    def __init__(self,dataset, train=True, inference=True, train_ratio=0.7, seed=42, random_coords = True):

        print('\nCreating Dataloader:')
        self.x = dataset.x
        self.Re = dataset.Re
        self.y = dataset.y
        self.patch_indices = dataset.patch_indices.copy()
        if dataset.bc is None:
            self.bc_values = None
            self.bc_coords = None
        else:
            self.bc_values = dataset.bc['Values'].clone()
            self.bc_coords = dataset.bc['Coords']

        self.random_coords = random_coords
        self.train = train
        self.data_splitter(train,inference,train_ratio,seed)

    def data_splitter(self, train, inference, train_ratio, seed):
        n_batches = self.y.shape[0]
        train_size = int(train_ratio * n_batches)
        test_size = n_batches - train_size
        if inference:
            seed_generator = torch.Generator().manual_seed(seed)

            # we want to pin the end points as training (to ensure complete inference)
            train_split,  test_split        = torch.utils.data.random_split(self.y[1:-1,...], [train_size-2, test_size], generator=seed_generator)
            test_split.indices = list(1 + np.array(test_split.indices))
            train_split.indices = list(1 + np.array(train_split.indices))
            train_split.indices.append(0)
            train_split.indices.append(-1)
        
            # The torch.utils.data.random_split() only gives objects with the whole datset or a integers, so we need to override these variables with the indexed datset split
            train_dataset,  test_dataset    = self.y[train_split.indices,...], self.y[test_split.indices,...]
            train_Re,    test_Re            = self.Re[train_split.indices], self.Re[test_split.indices]
            if self.bc_values is not None:
                train_bc, test_bc           = self.bc_values[train_split.indices,...], self.bc_values[test_split.indices,...]

            print(f'    Dataset Split up for inference using torch generator seed: {seed_generator.initial_seed()}')
        
        else:
            train_dataset,  test_dataset    = self.y[:train_size,...],  self.y[train_size:,...]
            train_Re,    test_Re            = self.Re[:train_size,...],     self.Re[train_size:,...]
            if self.bc_values is not None:
                train_bc, test_bc           = self.bc_values[:train_size,...], self.bc_values[train_size:,...]

            print(f'    Dataset Split up for High reynolds number extrapolation')

        if train:
            self.y = train_dataset
            self.Re = train_Re
            if self.bc_values is not None:
                self.bc_values = train_bc
            print(f'    Training Dataset Selected')
        else:
            self.y = test_dataset
            self.Re = test_Re
            if self.bc_values is not None:
                self.bc_values = test_bc
            print(f'    Testing Dataset Selected')

    def __len__(self):
        return len(self.Re)
    
    def __getitem__(self, idx):
        
        # randomize input coordinates
        if self.random_coords:
            indices = torch.randperm(self.x.shape[0])
            reverse_indices = torch.argsort(indices)
            reverse_patch_indices = dict.fromkeys(self.patch_indices)
            for patch in self.patch_indices:
                reverse_patch_indices[patch] = reverse_indices[self.patch_indices[patch]]
        else:
            indices = torch.arange(self.x.shape[0])
            reverse_indices = indices
            reverse_patch_indices = self.patch_indices.copy()
        
        in_keys     = self.Re[idx].float().reshape(1,1)
        in_queries  = self.x[indices,...].float()
        out_truth   = self.y[idx,indices,...].float()
        
        if self.bc_values is None:
            input_f = in_keys.float()
        else:
            in_bc_value = self.bc_values[idx,:,...].float()
            input_f = [in_keys, in_bc_value, self.bc_coords.float()]

        index_list = {'All Indices' : reverse_indices,
                      'Boundary Indices' :reverse_patch_indices
                      }
        
        return in_queries, input_f, out_truth, index_list