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
from train_utils.loss_functions import LP_custom, Linf_custom
from data_utils.utils import UnitTransformer, MultipleTensors
from data_utils.cavity_isotropic_subsampler import cavity_isotropic_subsampler

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
        y = output_normalizer.transform(y, inverse = True) # <- for dircihlet BC

    # Override (Hard Enforce) Boundaries (Only Velocity)
    # for patch in index['Boundary Indices']:#['movingWall']:
    #     out[:,index['Boundary Indices'][patch].flatten(),:2] = y[:,index['Boundary Indices'][patch].flatten(),:2]
    
    # Caclulate PDE and BC Losses

    # PDE
    x_i = keys_normalizer.transform(x_i, inverse = True)  
    pde_loss_list, derivatives = ns_pde_autograd_loss(x,out,Re=x_i,loss_function=loss_function)#,bc_index=index['Boundary Indices'])
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

    return loss_logger.aggregate(), out.detach().cpu().numpy(), x.detach().cpu().numpy()


if __name__ == '__main__':# 0. Get Arguments 
    
    dataset_args, model_args, training_args = get_default_args()
    dataset_args, model_args, training_args = args_override(dataset_args, model_args, training_args, ARGS)

    loaded_model_args = np.load(model_args['ckpt_path']+'_results.npy',allow_pickle=True).item()['Model Configuration']
    for setting in loaded_model_args:
        model_args[setting] = loaded_model_args[setting] 
    
    # Override for Step Case
    dataset_args['name'] = 'Cavity'
    # dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\backward_facing_step_normalized.npy'
    #dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\cavity_with_bc_normalized.npy'
    
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
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_args['step_size'], gamma=0.7)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=training_args['base_lr'], 
                                                        div_factor=1e4, 
                                                        pct_start=0.2, 
                                                        final_div_factor=1e4, 
                                                        steps_per_epoch=1, 
                                                        epochs=training_args['epochs']
                                                        )
    relobralo = RELOBRALO(device=device)

    loss_fn = LP_custom() #Linf_custom()  #torch.nn.MSELoss()
    
    intermediate_results = np.zeros((4,batch[2].shape[-2], batch[2].shape[-1]))
    intermediate_results_list = dict()
    intermediate_results_index = 0

    # 4. Train Epochs
    for epoch in range(training_args['epochs']):

        output_log, solution, input = fine_tune_case(model, 
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
        
        if (epoch+1) % (training_args['epochs']/4) == 0:
            intermediate_results[intermediate_results_index, ...] = solution # this appears to only work for the first time?

    intermediate_results_list['solutions'] = intermediate_results
    intermediate_results_list['coordinates'] = batch[0]
    train_logger.dictionary['intermediate_results'] = intermediate_results_list

    # Save model checkpoints
    save_checkpoint(training_args["save_dir"], 
                    training_args["save_name"] + f'_fine_RE{RE_number:.0f}', 
                    model=model, 
                    loss_dict=train_logger.dictionary, 
                    optimizer=optimizer) 

    