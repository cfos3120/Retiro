import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from data_utils.dataset_prep import create_loader, prepare_dataset
from data_utils.utils import get_seed
from models.retiro_model import GNOT
from models.old_model import CGPTNO
from train_utils.navier_stokes_autograd import ns_pde_autograd_loss, wrapped_model
from train_utils.boundary_conditions import bc_loss
from train_utils.logging import total_loss_list
from train_utils.loss_functions import LP_custom

from argparse import ArgumentParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
parser = ArgumentParser(description='GNOT (retiro) Artemis Training Study')
parser.add_argument('--data_path'   , type=str  , default= r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\GNOT\data\steady_cavity_case_b200_maxU100ms_simple_normalized.npy')
parser.add_argument('--sub_x'       , type=float, default=8)
parser.add_argument('--seed'        , type=int  , default=42)
parser.add_argument('--rand_cood'   , type=int  , default=1)
parser.add_argument('--case_path'   , type=str  , default='None')
parser.add_argument('--name'        , type=str  , default='test')

ARGS = parser.parse_args()


torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)
get_seed(ARGS.seed, printout=True, cudnn=False)
seed_generator = torch.Generator().manual_seed(42)

if __name__ == '__main__':

    # 0. Get Arguments
    #case_name_path = r'Z:\PRJ-MLFluids\Retiro_results\test_step_v1\test_aMonitor_b0_c1_d0.2_h200'
    case_name_path = ARGS.case_path
    results_file = np.load(case_name_path+'_results.npy',allow_pickle=True).item()
    model_args = results_file['Model Configuration']
    dataset_args = results_file['Data Configuration']

    # Override for Step Case
    #dataset_args['name'] = 'Step'
    #dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\backward_facing_step_normalized.npy'
    #dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\cavity_with_bc_normalized.npy'
    
    dataset_args['rand_cood'] = ARGS.random_coords == 1
    dataset_args['file_path'] = ARGS.data_path
    dataset_args['sub_x'] = ARGS.sub_x

    # 1. Dataset Preperations:
    dataset = prepare_dataset(dataset_args, unsupervised=False)
    torch_dataset = create_loader(dataset, dataset_args)
    train_loader = DataLoader(
                            dataset=torch_dataset,
                            batch_size=1,
                            shuffle=True,
                            generator=seed_generator
                        )
    dataset_args['train'] = False
    torch_dataset_2 = create_loader(dataset, dataset_args)
    eval_loader = DataLoader(
                            dataset=torch_dataset_2,
                            batch_size=1,
                            shuffle=True,
                            generator=seed_generator
                        )
    
    # 2. Initialize Model:
    model = CGPTNO(branch_sizes=torch_dataset.config['branch_sizes'],
                    n_layers=model_args['n_layers'],
                    n_hidden=model_args['n_hidden'])

    model = wrapped_model(model=model,
                          query_normalizer=dataset.x_normalizer.to(device),
                          )
    
    ckpt = torch.load(case_name_path+'.pt', map_location=device)

    model.load_ckpt(ckpt['model'])
    print('Weights loaded from %s' % case_name_path)

    model = model.to(device)

    # 3. Validation Hyperparameters:
    train_logger = total_loss_list(model_config=model_args, data_config=dataset_args)
    
    loss_fn_1 = LP_custom()
    loss_fn_2 = LP_custom()

    output_normalizer = dataset.y_normalizer.to(device)
    keys_normalizer = dataset.xi_normalizer.to(device)
    
    model.eval()
    
    data_labels = ['Training Set', 'Validation Set']
    for i, data_loader in enumerate([train_loader, eval_loader]):
        for x, x_i, y, index in data_loader:
            x, x_i, y = x.to(device), x_i.to(device), y.to(device)
            x.requires_grad = True
            
            # infer model
            t0 = time.time()
            out = model(x, inputs=x_i)
            t1 = time.time()

            # Un-normalize output if not already in wrapped model forward()
            if model.output_normalizer is None:
                out = output_normalizer.transform(out, inverse = True)
                y = output_normalizer.transform(y, inverse = True)       
            
            supervised_loss = loss_fn_1(out,y)

            # PDE
            x_i = keys_normalizer.transform(x_i, inverse = True)  
            pde_loss_list, derivatives = ns_pde_autograd_loss(x,out,Re=x_i,loss_function=loss_fn_2)

            # BC (von Neumann and Dirichlet)
            bc_loss_list = bc_loss(out,y,bc_index=index['Boundary Indices'],derivatives=derivatives,loss_function=loss_fn_2)

            loss_dict = {'Re Number' : round(x_i.flatten().item(),2),
                        'L2 Rel Loss' : supervised_loss.item(),
                        'PDE 1 Abs Loss' : pde_loss_list[0].item(),
                        'PDE 2 Abs Loss' : pde_loss_list[1].item(),
                        'PDE 3 Abs Loss' : pde_loss_list[2].item(),
                        'BC D Abs Loss' : bc_loss_list[0].item(),
                        'BC VN Abs Loss' : bc_loss_list[1].item(),
                        'Inference Time' : t1-t0,
                        'Label': data_labels[i]
                        }

            train_logger.update(loss_dict)

    np.save(case_name_path+'_l2_loss',train_logger.fetch_dict(),allow_pickle=True)