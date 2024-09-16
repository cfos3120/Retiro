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

class pinn_zeroShot():
    def __init__(self, model, case, keys_normalizer, output_normalizer, loss_function=torch.nn.MSELoss()):
        self.model = model
        self.x = case[0]
        self.xi = case[1]
        self.y = case[2]
        self.x, self.xi, self.y = self.x.to(device), self.xi.to(device), self.y.to(device)
        self.output_normalizer = output_normalizer
        self.y = self.output_normalizer.transform(self.y, inverse = True) # <- for dircihlet BC
        self.keys_normalizer = keys_normalizer
        self.bc_index = case[3]

        self.x.requires_grad = True

        self.loss_function = loss_function
        self.optimizer = torch.optim.LBFGS(self.model.parameters(),
                                    lr=1,
                                    max_iter=50000,
                                    max_eval=50000,
                                    history_size=50,
                                    tolerance_grad=1e-05,
                                    tolerance_change=0.5 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")
        
        self.iter = 0
        

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u & f predictions:
        all_losses_list = []

        out = model(self.x, inputs=self.xi)
        out = self.output_normalizer.transform(out, inverse = True)
        supervised_loss = self.loss_function(out[...],self.y[...])

        # PDE
        xi = self.keys_normalizer.transform(self.xi, inverse = True)  
        pde_loss_list, derivatives = ns_pde_autograd_loss(self.x,out,Re=xi,loss_function=self.loss_function,
                                                        pressure=True, 
                                                        bc_index=self.bc_index['Boundary Indices']
                                                        )
        all_losses_list += pde_loss_list

        # BC (von Neumann and Dirichlet)
        bc_loss_list = bc_loss(out,self.y,bc_index=self.bc_index['Boundary Indices'],
                               derivatives=derivatives,
                               loss_function=self.loss_function,
                               ARGS=ARGS)
        all_losses_list += bc_loss_list
        

        self.ls = 0
        for loss in all_losses_list:
            self.ls += loss

        self.ls = self.ls.float()

        # Store losses:
        loss_logger = loss_aggregator()
        keys = ['PDE 1 (c)', 'PDE 2 (x)', 'PDE 3 (y)', 'PDE 4 (p)', 'BC (D)', 'BC (VN)']
        loss_dict = {keys[i]:j.item() for i,j in enumerate(all_losses_list)}
        loss_dict.update({'Total_Loss':self.ls.item()})
        loss_dict.update({'Supervised Loss':supervised_loss.item()})
        loss_logger.add(loss_dict)

        model.train_logger.update(loss_logger.aggregate())
        # derivative with respect to net's weights:
        self.ls.backward()

        # increase iteration count:
        self.iter += 1

        # print report:
        if not self.iter % 100:
            print('Epoch: {0:}, Loss: {1:6.3f}'.format(self.iter, self.ls))

        return self.ls
    
    def train(self):
        """ training loop """
        self.model.train()
        self.optimizer.step(self.closure)

if __name__ == '__main__':# 0. Get Arguments 
    
    dataset_args, model_args, training_args = get_default_args()
    dataset_args, model_args, training_args = args_override(dataset_args, model_args, training_args, ARGS)

    loaded_model_args = np.load(model_args['ckpt_path']+'_results.npy',allow_pickle=True).item()['Model Configuration']
    for setting in loaded_model_args:
        model_args[setting] = loaded_model_args[setting] 
    
    # Override for Step Case
    dataset_args['name'] = 'Cavity'
    # dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\backward_facing_step_normalized.npy'
    # dataset_args['file_path'] = r'C:\Users\Noahc\Documents\USYD\tutorial\python_utils\cavity_with_bc_normalized.npy'
    
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
                          #output_normalizer=dataset.y_normalizer.to(device)
                          )
    model = model.to(device)

    # Load checkpoint
    ckpt = torch.load(model_args['ckpt_path']+'.pt', map_location=device)
    model.load_ckpt(ckpt['model'])

    # 3. Training Hyperparameters:
    model.train_logger = total_loss_list(model_config=model_args, training_config=training_args, data_config=dataset_args)
    
    zeroShot_model = pinn_zeroShot(model, batch, 
                                   keys_normalizer=dataset.xi_normalizer.to(device), 
                                   output_normalizer=dataset.y_normalizer.to(device))

    zeroShot_model.train()

    # Save model checkpoints
    save_checkpoint(training_args["save_dir"], 
                    training_args["save_name"] + f'_fine_RE{RE_number:.0f}', 
                    model=zeroShot_model.model, 
                    loss_dict=zeroShot_model.model.train_logger.dictionary)