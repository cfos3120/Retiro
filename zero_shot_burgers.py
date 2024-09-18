import torch
from torch.utils.data import DataLoader
import numpy as np
import itertools
from random import uniform

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

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
get_seed(42, printout=True, cudnn=False)
seed_generator = torch.Generator().manual_seed(42)

def grid_burger():
    N_u = 100                 # number of data points in the boundaries
    N_f = 10000               # number of collocation points

    # X_u_train: a set of pairs (x, t) located at:
        # x =  1, t = [0,  1]
        # x = -1, t = [0,  1]
        # t =  0, x = [-1, 1]
    x_upper = np.ones((N_u//4, 1), dtype=float)
    x_lower = np.ones((N_u//4, 1), dtype=float) * (-1)
    t_zero = np.zeros((N_u//2, 1), dtype=float)

    t_upper = np.random.rand(N_u//4, 1)
    t_lower = np.random.rand(N_u//4, 1)
    x_zero = (-1) + np.random.rand(N_u//2, 1) * (1 - (-1))

    # stack uppers, lowers and zeros:
    X_upper = np.hstack( (x_upper, t_upper) )
    X_lower = np.hstack( (x_lower, t_lower) )
    X_zero = np.hstack( (x_zero, t_zero) )

    # each one of these three arrays haS 2 columns, 
    # now we stack them vertically, the resulting array will also have 2 
    # columns and 100 rows:
    X_u_train = np.vstack( (X_upper, X_lower, X_zero) )

    # shuffle X_u_train:
    index = np.arange(0, N_u)
    np.random.shuffle(index)
    X_u_train = X_u_train[index, :]
    
    # make X_f_train:
    X_f_train = np.zeros((N_f, 2), dtype=float)
    for row in range(N_f):
        x = uniform(-1, 1)  # x range
        t = uniform( 0, 1)  # t range

        X_f_train[row, 0] = x 
        X_f_train[row, 1] = t

    # add the boundary points to the collocation points:
    X_f_train = np.vstack( (X_f_train, X_u_train) )

    # make u_train
    u_upper =  np.zeros((N_u//4, 1), dtype=float)
    u_lower =  np.zeros((N_u//4, 1), dtype=float) 
    u_zero = -np.sin(np.pi * x_zero)  

    # stack them in the same order as X_u_train was stacked:
    u_train = np.vstack( (u_upper, u_lower, u_zero) )

    # match indices with X_u_train
    u_train = u_train[index, :]

    return u_train, X_f_train, X_u_train

def ns_burgers(model_input_coords, model_out, Re):

    # Stack and Repeat Re for tensor multiplication
    Re = Re.squeeze(-1)
    #print('Re', Re.item(), "Max:", model_input_coords.max().item(), "Min:",model_input_coords.min().item())

    u = model_out

    # First Derivatives
    u_x = torch.autograd.grad(u, model_input_coords, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][...,0]
    u_t = torch.autograd.grad(u, model_input_coords, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][...,1]

    # Second Derivatives
    u_xx = torch.autograd.grad(u_x, model_input_coords, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][...,0]

    
    # Navier-Stokes equation
    f1 = u_t + u*u_x - ((1/Re)*u_xx)
    #print(torch.mean(u_x), torch.mean(u_t), torch.mean(u_xx), torch.mean(f1))
    
    return f1


class pinn_zeroShot():
    def __init__(self, model, 
                X_f_train,       
                Re,
                X_u_train,
                u_train,
                keys_normalizer=None, output_normalizer=None, loss_function=torch.nn.MSELoss()):
        self.model = model
        self.x = X_f_train.to(device)
        self.x2 = X_u_train.to(device)
        self.xi = Re.to(device)
        self.y2 = u_train.to(device)

        self.output_normalizer = output_normalizer
        #self.y = self.output_normalizer.transform(self.y, inverse = True) # <- for dircihlet BC
        self.keys_normalizer = keys_normalizer

        self.x.requires_grad = True
        self.x2.requires_grad = True

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
        self.ls = 0
        

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        out_all = model(self.x, inputs=self.xi)
        out_bc = model(self.x2, inputs=self.xi)

        #out = self.output_normalizer.transform(out, inverse = True)

        # PDE
        #xi = self.keys_normalizer.transform(self.xi, inverse = True)  
        f1 = ns_burgers(self.x, out_all, self.xi)

        loss_pde = self.loss_function(f1, torch.zeros_like(f1))
        loss_bc = self.loss_function(out_bc, self.y2)
        
        self.ls = loss_pde + loss_bc

        # derivative with respect to net's weights:
        self.ls.backward()

        # increase iteration count:
        self.iter += 1

        # print report:
        if not self.iter % 100:
            print('Epoch: {0:}, Loss: {1:9.5f} Full Loss {1:9.5f} Boundary Loss {1:9.5f}'.format(self.iter, self.ls, loss_pde.item(), loss_bc.item()))

        return self.ls
    
    def train(self):
        """ training loop """
        self.model.train()
        self.optimizer.step(self.closure)

if __name__ == '__main__':# 0. Get Arguments 
    
    Re = 820
    # 2. Initialize Model:
    
    model = CGPTNO(branch_sizes=[1],
                    output_size = 1,
                    n_layers=2,
                    n_hidden=20) #GNOT(n_experts=1) # #

    model = wrapped_model(model=model,
                          #query_normalizer=dataset.x_normalizer.to(device),
                          #output_normalizer=dataset.y_normalizer.to(device)
                          )
    model = model.to(device)

    # 3. Training Hyperparameters:

    u_train, X_f_train, X_u_train= grid_burger()
    u_train = torch.tensor(u_train,dtype=float).unsqueeze(0)
    X_f_train = torch.tensor(X_f_train,dtype=float).unsqueeze(0)
    X_u_train = torch.tensor(X_u_train,dtype=float).unsqueeze(0)
    Re = torch.tensor(Re,dtype=float).reshape(1,1,1)


    print(u_train.shape, X_f_train.shape, X_u_train.shape)
    zeroShot_model = pinn_zeroShot(model, 
                                   X_f_train.float(),
                                   Re.float(),
                                   X_u_train.float(),
                                   u_train.float())
    #                                #keys_normalizer=dataset.xi_normalizer.to(device), 
    #                                #output_normalizer=dataset.y_normalizer.to(device)
    #                                )

    #print(zeroShot_model.closure())
    zeroShot_model.train()

    # # Save model checkpoints
    save_checkpoint("test_gnot_burgers", 
                    f'example_fine_RE{Re.item():.0f}', 
                    model=zeroShot_model.model)