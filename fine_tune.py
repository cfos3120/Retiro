import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools

from data_utils.dataset_prep import create_loader, prepare_dataset
from data_utils.utils import parse_arguments, get_seed
from models.retiro_model import GNOT
from models.old_model import CGPTNO
from train_utils.navier_stokes_autograd import ns_pde_autograd_loss, wrapped_model
from train_utils.boundary_conditions import bc_loss
from train_utils.dynamic_loss_balancing import RELOBRALO
from train_utils.logging import loss_aggregator, total_loss_list, save_checkpoint
from train_utils.default_args import get_default_args, args_override 
from train_utils.loss_functions import LP_custom
from data_utils.utils import UnitTransformer, MultipleTensors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

global ARGS 
ARGS = parse_arguments()


torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)
get_seed(ARGS.seed, printout=True, cudnn=False)
seed_generator = torch.Generator().manual_seed(42)

def fine_tune_case(model, case, optimizer, loss_function=torch.nn.MSELoss(), output_normalizer=None, keys_normalizer=None, dyn_loss_bal = False):

    keys = ['PDE 1 (c)', 'PDE 2 (x)', 'PDE 3 (y)', 'BC (D)', 'BC (VN)']
    loss_logger = loss_aggregator()

    x, x_i, y, index = case #(case is a batch)
    x, x_i, y = x.to(device), x_i.to(device), y.to(device)
    x.requires_grad = True

    all_losses_list = []

    # infer model
    out = model(x, inputs=x_i)

    # Calculate supervised pointwise loss (for validation only)
    supervised_loss = loss_function(out,y)

    # Un-normalize output if not already in wrapped model forward()
    if model.output_normalizer is None:
        out = output_normalizer.transform(out, inverse = True)          
    
    # Caclulate PDE and BC Losses

    # PDE
    x_i = keys_normalizer.transform(x_i, inverse = True)  
    pde_loss_list, derivatives = ns_pde_autograd_loss(x,out,Re=x_i,loss_function=loss_function)
    all_losses_list += pde_loss_list

    # BC (von Neumann and Dirichlet)
    bc_loss_list = bc_loss(out,y,bc_index=index['Boundary Indices'],derivatives=derivatives,loss_function=loss_function,ARGS=ARGS)
    all_losses_list += bc_loss_list
    total_losses_bal = relobralo(loss_list=all_losses_list) + 0.0*supervised_loss

    # Store losses:
    loss_dict = {keys[i]:j.item() for i,j in enumerate(all_losses_list)}
    loss_dict.update({'Total_Loss':total_losses_bal.item()})
    loss_dict.update({'Supervised Loss':supervised_loss.item()})
    loss_dict.update({i:relobralo.lam[i].item() for i in relobralo.lam.keys()})
    loss_logger.add(loss_dict)

    # Update model
    total_losses_bal.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
    optimizer.step()
    optimizer.zero_grad()

    return loss_logger.aggregate()

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

if __name__ == '__main__':# 0. Get Arguments 
    
    dataset_args, model_args, training_args = get_default_args()
    dataset_args, model_args, training_args = args_override(dataset_args, model_args, training_args, ARGS)

    loaded_model_args = np.load(model_args['ckpt_path']+'_results.npy',allow_pickle=True).item()['Model Configuration']
    for setting in loaded_model_args:
        model_args[setting] = loaded_model_args[setting] 
    
    # Override for Step Case
    dataset_args['name'] = 'Step'
    # dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\backward_facing_step_normalized.npy'
    dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\cavity_with_bc_normalized.npy'
    
    # hard override so we can use the sub_x args.
    sub_x = int(ARGS.sub_x)
    dataset_args['sub_x'] = 1
    dataset_args['random_coords'] = False


    # 1. Dataset Preperations:
    dataset_args['train'] = False
    dataset = prepare_dataset(dataset_args, unsupervised=False)
    torch_dataset = create_loader(dataset, dataset_args)
    train_loader = DataLoader(
                            dataset=torch_dataset,
                            batch_size=1,
                            shuffle=False
                        )
    
    case_n = training_args['case_n']
    batch = next(itertools.islice(train_loader, case_n, None))
    batch[0], batch[2], batch[3] = cavity_isotropic_subsampler(batch[0],batch[2],batch[3], sub_x)

    RE_number = dataset.xi_normalizer.transform(batch[1], inverse = True).item()
    print(f'\nFine Tuning Case: RE {RE_number:.1f}\n')

    # 2. Initialize Model:
    
    model = CGPTNO(branch_sizes=torch_dataset.config['branch_sizes'],
                    n_layers=model_args['n_layers'],
                    n_hidden=model_args['n_hidden']) #GNOT(n_experts=1) # #

    if model_args['init_w']:
        model.apply(model._init_weights)

    if training_args['DP']:
       model = torch.nn.DataParallel(model)
    
    model = wrapped_model(model=model,
                          query_normalizer=dataset.x_normalizer.to(device),
                          )
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(model_args['ckpt_path']+'.pt', map_location=device)
    model.load_ckpt(ckpt['model'])

    # 3. Training Hyperparameters:
    train_logger = total_loss_list(model_config=model_args, training_config=training_args, data_config=dataset_args)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                    betas=(0.9, 0.999), 
                                    lr=training_args['base_lr'],
                                    weight_decay=training_args['weight-decay']
                                    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_args['step_size'], gamma=0.7)
    relobralo = RELOBRALO(device=device)

    loss_fn = LP_custom() #torch.nn.MSELoss()
    
    # 4. Train Epochs
    for epoch in range(training_args['epochs']):

        output_log = fine_tune_case(model, 
                                    batch, 
                                    optimizer, 
                                    loss_function=loss_fn,
                                    output_normalizer=dataset.y_normalizer.to(device),
                                    keys_normalizer=dataset.xi_normalizer.to(device),
                                    dyn_loss_bal = training_args['dynamic_balance']
                                    )
        train_logger.update(output_log)
        scheduler.step()
        
        print(f"[Epoch{epoch:5.0f}] Loss: {output_log['Total_Loss']:.4E} | " + \
              f"Supervised Loss: {output_log['Supervised Loss']:.4E} | " + \
              f"PDE 1 (c): {output_log['PDE 1 (c)']:.4E} | " + \
              f"PDE 2 (x): {output_log['PDE 2 (x)']:.4E} | " + \
              f"PDE 3 (y): {output_log['PDE 3 (y)']:.4E} | " + \
              f"BC (D): {output_log['BC (D)']:.4E} | " + \
              f"BC (VN): {output_log['BC (VN)']:.4E}"
              )

    # Save model checkpoints
    save_checkpoint(training_args["save_dir"], 
                    training_args["save_name"] + f'_fine_RE{RE_number:.0f}', 
                    model=model, 
                    loss_dict=train_logger.dictionary, 
                    optimizer=optimizer) 

    