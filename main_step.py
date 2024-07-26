import torch
from torch.utils.data import DataLoader
import numpy as np

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


def hybrid_train_batch(model, dataloader, optimizer, loss_function=torch.nn.MSELoss(), hybrid_type = 'Train', output_normalizer=None, keys_normalizer=None, dyn_loss_bal = False):
    l_nu = 10 # (reynolds number multiplier L/nu * lid velocity)

    if hybrid_type in ['Train','Monitor']:
        assert (model.output_normalizer is not None or output_normalizer is not None), \
            'Output normalizer needs to be defined in function if Wrapped model does not feature output normalizer in forward()'
        assert keys_normalizer is not None, 'Keys need to be un-normalized for PDE calculation'

    if dyn_loss_bal and hybrid_type == 'Train': 
        relobralo = RELOBRALO(device=device) 

    keys = ['Supervised Loss', 'PDE 1 (c)', 'PDE 2 (x)', 'PDE 3 (y)', 'BC (D)', 'BC (VN)']
    loss_logger = loss_aggregator()

    for x, x_i, y, index in dataloader:
        #x, x_i, y = x.to(device), MultipleTensors(x_i).to(device), y.to(device)
        x, x_i, y = x.to(device), x_i.to(device), y.to(device)
        x.requires_grad = True

        optimizer.zero_grad()
        all_losses_list = []
        
        # infer model
        out = model(x, inputs=x_i)

        # Calculate supervised pointwise loss
        supervised_loss = loss_function(out,y)
        all_losses_list += [supervised_loss]

        # Un-normalize output if not already in wrapped model forward()
        if model.output_normalizer is None:
            out = output_normalizer.transform(out, inverse = True)          
        
        # Caclulate PDE and BC Losses
        if hybrid_type in ['Train','Monitor']:

            # PDE
            x_i = keys_normalizer.transform(x_i, inverse = True)  
            pde_loss_list, derivatives = ns_pde_autograd_loss(x,out,Re=x_i*l_nu,loss_function=loss_function)
            all_losses_list += pde_loss_list

            # BC (von Neumann and Dirichlet)
            bc_loss_list = bc_loss(out,y,bc_index=index['Boundary Indices'],derivatives=derivatives,loss_function=loss_function)
            all_losses_list += bc_loss_list

        # Balance Losses
        if dyn_loss_bal and hybrid_type == 'Train':    
            total_losses_bal = relobralo(loss_list=all_losses_list)         # Dynamic Balance Losses
        elif hybrid_type == 'Train':               
            total_losses_bal = sum(all_losses_list)/len(all_losses_list)    # Simply Mean Losses
        else:
            total_losses_bal = supervised_loss                              # Only backwards bass supervised loss
        
        # Store losses:
        loss_dict = {keys[i]:j.item() for i,j in enumerate(all_losses_list)}
        loss_dict.update({'Total_Loss':total_losses_bal.item()})
        if dyn_loss_bal and hybrid_type == 'Train':
            loss_dict.update({i:relobralo.lam[i].item() for i in relobralo.lam.keys()})

        loss_logger.add(loss_dict)

        # Update model
        total_losses_bal.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
        optimizer.step()
    
    return loss_logger.aggregate()


def unsupervised_train(model, dataloader, optimizer, output_normalizer=None, keys_normalizer=None, dyn_loss_bal=False,loss_function=torch.nn.MSELoss()):

    l_nu = 10 # (reynolds number multiplier L/nu * lid velocity)

    assert model.output_normalizer is None and output_normalizer is not None, 'Output normalizer needs to be defined in function if'\
            ' Wrapped model does not feature output normalizer in forward()'

    if dyn_loss_bal: 
        relobralo = RELOBRALO(device=device) 

    keys = ['Unsupervised PDE 1 (c)', 'Unsupervised PDE 2 (x)', 'Unsupervised PDE 3 (y)']
    loss_logger = loss_aggregator()

    for x, x_i,__ in dataloader:
        all_losses_list = []
        x, x_i = x.to(device), x_i.to(device)
        x.requires_grad = True

        optimizer.zero_grad()

        # infer model
        out = model(x, inputs=x_i)

        # Un-normalize output if not already in wrapped model forward()
        if model.output_normalizer is None:
            out = output_normalizer.transform(out, inverse = True)          
        
        # Caclulate PDE Losses
        x_i = keys_normalizer.transform(x_i, inverse = True)  
        pde_loss_list   = ns_pde_autograd_loss(x,out,Re=x_i*l_nu,loss_function=loss_function)
        all_losses_list += pde_loss_list

        # Balance Losses
        if dyn_loss_bal:    
            total_losses_bal = relobralo(loss_list=all_losses_list)         # Dynamic Balance Losses
        else:      
            total_losses_bal = sum(all_losses_list)/len(all_losses_list)    # Simply Mean Losses
        
        # Store losses:
        loss_dict = {keys[i]:j.item() for i,j in enumerate(all_losses_list)}
        loss_dict.update({'Unsupervised Total_Loss':total_losses_bal.item()})
        if dyn_loss_bal:
            loss_dict.update({'Unsupervised ' +i:relobralo.lam[i].item() for i in relobralo.lam.keys()})
        loss_logger.add(loss_dict)

        # Update model
        total_losses_bal.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000)
        optimizer.step()
        
    
    return loss_logger.aggregate()


def validation(model, dataloader, loss_function):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x, x_i, y,__ in dataloader:
            x, x_i, y = x.to(device), x_i.to(device), y.to(device)
            
            out = model(x,inputs=x_i)      
            
            val_loss += loss_function(out, y).item()

        val_loss = val_loss/len(dataloader)

    return {'Validation Loss' :val_loss}


if __name__ == '__main__':

    # 0. Get Arguments 
    dataset_args, model_args, training_args = get_default_args()
    dataset_args, model_args, training_args = args_override(dataset_args, model_args, training_args, ARGS)

    # Override for Step Case
    #dataset_args['name'] = 'Step'
    dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\backward_facing_step_normalized.npy'
    dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\cavity_with_bc_normalized.npy'
    dataset_args['sub_x'] = 0.001
    # dataset_args['bc_input_f'] = False

    # 1. Dataset Preperations:
    dataset = prepare_dataset(dataset_args, unsupervised=False)
    torch_dataset = create_loader(dataset, dataset_args)
    train_loader = DataLoader(
                            dataset=torch_dataset,
                            batch_size=dataset_args['batchsize'],
                            shuffle=True,
                            generator=seed_generator
                        )

    if training_args['Key_only_batches'] > 0:
        keys_only_dataset = prepare_dataset(dataset_args, unsupervised=True, reference_data_set=dataset)
        keys_only_torch_dataset = create_loader(keys_only_dataset, dataset_args)
        keys_only_train_loader = DataLoader(
                                            dataset=keys_only_torch_dataset,
                                            batch_size=dataset_args['batchsize'],
                                            shuffle=True,
                                            generator=seed_generator
                                        )
    
    if training_args['eval_while_training']:
        dataset_args['train'] = False
        val_torch_dataset = create_loader(dataset, dataset_args)
        val_loader = DataLoader(
                                dataset=val_torch_dataset,
                                batch_size=1,
                                shuffle=False,
                                generator=seed_generator
                            )

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


    # 3. Training Hyperparameters:
    train_logger = total_loss_list(model_config=model_args, training_config=training_args, data_config=dataset_args)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                    betas=(0.9, 0.999), 
                                    lr=training_args['base_lr'],
                                    weight_decay=training_args['weight-decay']
                                    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_args['step_size'], gamma=0.7)

    if training_args['Secondary_optimizer']:
        optimizer2 = torch.optim.AdamW(model.parameters(), 
                                    betas=(0.9, 0.999), 
                                    lr=training_args['base_lr'],
                                    weight_decay=training_args['weight-decay']
                                    )
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=training_args['step_size'], gamma=0.7)
        
    loss_fn = LP_custom() #torch.nn.MSELoss()
    
    # 4. Train Epochs
    for epoch in range(training_args['epochs']):

        output_log = hybrid_train_batch(model, 
                                        train_loader, 
                                        optimizer, 
                                        loss_function=loss_fn, 
                                        hybrid_type=training_args['Hybrid_type'], 
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
        
        if training_args['Key_only_batches'] > 0:
            output_log = unsupervised_train(model, 
                                            keys_only_train_loader, 
                                            optimizer = optimizer2 if training_args['Secondary_optimizer'] else optimizer, 
                                            output_normalizer=dataset.y_normalizer.to(device), 
                                            keys_normalizer=dataset.xi_normalizer.to(device),
                                            dyn_loss_bal = training_args['dynamic_balance'],
                                            loss_function=loss_fn
                                            )
            train_logger.update(output_log)
            if training_args['Secondary_optimizer']:
                scheduler2.step()
            
        if training_args['eval_while_training']:
            output_log = validation(model, val_loader, loss_function=loss_fn)
            train_logger.update(output_log)

    # Save model checkpoints
    save_checkpoint(training_args["save_dir"], 
                    training_args["save_name"], 
                    model=model, 
                    loss_dict=train_logger.dictionary, 
                    optimizer=optimizer) 

    