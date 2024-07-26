from .cavity_full_dataset import Cavity_2D_dataset_for_GNOT, CavityDataset
from .cavity_keys_only_dataset import unsupervised_Cavity_dataset_for_GNOT, unsupervised_CavityDataset

from .step_full_dataset import Step_2D_dataset_for_GNOT, StepDataset
from .step_keys_only_dataset import unsupervised_Step_dataset_for_GNOT, unsupervised_StepDataset

def prepare_dataset(args, unsupervised=False, reference_data_set=None):
    
    # need to add cavity or step option here:

    if args['name'] == 'Cavity (old Method)':
        if unsupervised:
            torch_dataset = unsupervised_Cavity_dataset_for_GNOT(key_range_min = args['key_range_min'], 
                                                        key_range_max = args['key_range_max'],
                                                        key_range_interval = 1,
                                                        L=args['L'],
                                                        resolution = args['keys only resolution'],
                                                        normalize_y=args['normalize_y'], 
                                                        normalize_x=args['normalize_x'], 
                                                        normalize_f=args['normalize_f'], 
                                                        vertex = args['vertex'],
                                                        boundaries=args['boundaries'],
                                                        reference_data_set=reference_data_set
                                                        )
        else:
            torch_dataset = Cavity_2D_dataset_for_GNOT(data_path=args['file_path'], 
                                                        L=args['L'], 
                                                        sub_x=args['sub_x'], # needs to be a factor
                                                        normalize_y=args['normalize_y'], 
                                                        normalize_x=args['normalize_x'], 
                                                        normalize_f=args['normalize_f'], 
                                                        vertex=args['vertex'], 
                                                        boundaries=args['boundaries']
                                                        )
    
    else:
        torch_dataset = Step_2D_dataset_for_GNOT(data_path=args['file_path'],
                                                        subsample_ratio=args['sub_x'], # needs to be a percentage
                                                        normalize_y=args['normalize_y'], 
                                                        normalize_x=args['normalize_x'], 
                                                        normalize_f=args['normalize_f'], 
                                                        boundary_input_f=args['bc_input_f']
                                                        )
        
    return torch_dataset
        


def create_loader(dataset, args):
    if dataset.__class__.__name__ == 'Cavity_2D_dataset_for_GNOT':
        dataloader = CavityDataset(dataset=dataset,
                                   train=args['train'], 
                                   inference=args['inference'],
                                   train_ratio=args['train_ratio'], 
                                   seed=args['seed'], 
                                   random_coords = args['random_coords'])

    elif dataset.__class__.__name__  == 'unsupervised_Cavity_dataset_for_GNOT':
        dataloader = unsupervised_CavityDataset(dataset=dataset,random_coords = args['random_coords'])

    elif dataset.__class__.__name__ == 'Step_2D_dataset_for_GNOT':
        dataloader = StepDataset(dataset=dataset,
                                   train=args['train'], 
                                   inference=args['inference'],
                                   train_ratio=args['train_ratio'], 
                                   seed=args['seed'], 
                                   random_coords = args['random_coords'])
    else:
        raise NotImplementedError('dataset is not supported')
    
    dataloader.config = dataset.config
    return dataloader

# Testing function
if __name__ == '__main__':

    for a in [True, False]:
        for b in [True, False]:
            for c in [True, False]:
                for d in [True, False]:
                    for e in [True, False]:
                        for f in [True, False]:   
                            for g in [True, False]:
                                for h in [True, False]:
                                    supervised_args = {'file_path':r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy',
                                                    'L':1.0,
                                                    'sub_x':8,
                                                    'normalize_y':a,
                                                    'normalize_x':b,
                                                    'normalize_f':c,
                                                    'vertex':d,
                                                    'boundaries':e,
                                                    'train':f,
                                                    'inference':g,
                                                    'random_coords':h}

                                    dataset = prepare_dataset(supervised_args, unsupervised=False)
                                    data_loader = create_loader(dataset, supervised_args)
                
    print('All function checks for supervised dataset passed')

    for a in [True, False]:
        for b in [True, False]:
            for c in [True, False]:
                for d in [True, False]:
                    for e in [True, False]:
                        for f in [None, dataset]:
                            for g in [True, False]:
                                unsupervised_args = {'key_range_min':1.0,
                                                    'key_range_max':100.0,
                                                    'key_range_interval':1,
                                                    'L':1.0,
                                                    'resolution':32,
                                                    'normalize_y':a,
                                                    'normalize_x':b,
                                                    'normalize_f':c,
                                                    'vertex':d,
                                                    'boundaries':e,
                                                    'reference_data_set':f,
                                                    'random_coords':g
                                                    }
                                dataset = prepare_dataset(unsupervised_args, unsupervised=True)
                                data_loader = create_loader(dataset, unsupervised_args)
    
    print('All function checks for unsupervised dataset passed')