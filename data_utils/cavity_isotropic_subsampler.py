import torch
import numpy as np

def cavity_isotropic_subsampler(x, y, index, factor=1):
    '''
    NOTE this function has inplace editing of the index list.
    Can only use it once for a given index.
    '''

    # cut out boundaries (not to be averaged)
    patch_mins = []
    for patch in index['Boundary Indices']:
        patch_mins += [index['Boundary Indices'][patch].min()]

    start_bc = np.min(patch_mins)
    x_no_bc = x[:,:start_bc,:]
    y_no_bc = y[:,:start_bc,:]
    
    # subsample through cell averaging
    B = x_no_bc.shape[0]
    S = int(np.sqrt(x_no_bc.shape[1]))
    C = x_no_bc.shape[-1]
    C2 = y_no_bc.shape[-1]

    x_sampled_no_bc = torch.nn.functional.avg_pool2d(x_no_bc.reshape(B,S,S,C).permute(0,3,1,2), factor).permute(0,2,3,1)
    y_sampled_no_bc = torch.nn.functional.avg_pool2d(y_no_bc.reshape(B,S,S,C2).permute(0,3,1,2), factor).permute(0,2,3,1)
    
    print(f'Dataset Subsampled by factor of {factor} with new shape {x_sampled_no_bc.shape}')
    x_sampled_no_bc = x_sampled_no_bc.reshape(B,x_sampled_no_bc.shape[1]**2,C)
    y_sampled_no_bc = y_sampled_no_bc.reshape(B,y_sampled_no_bc.shape[1]**2,C2)
    
    # add back boundary cells
    x_sampled = torch.concat([x_sampled_no_bc,x[:,start_bc:,:]],dim=1)
    y_sampled = torch.concat([y_sampled_no_bc,y[:,start_bc:,:]],dim=1)

    # correct index dict
    new_indices = index.copy()
    
    index_dif = start_bc - x_sampled_no_bc.shape[1]

    new_indices['All Indices'] = torch.arange(x_sampled.shape[1],dtype=int).unsqueeze(0).repeat(B,1)
    for patch in new_indices['Boundary Indices']:
        new_indices['Boundary Indices'][patch] = index['Boundary Indices'][patch] - index_dif 
        
    return x_sampled, y_sampled, new_indices
