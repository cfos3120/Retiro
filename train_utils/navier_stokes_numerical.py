import torch
import numpy as np

# Autograd Calculation of Gradients and Navier-Stokes Equations
def ns_pde_numerical(model_input_coords, model_out_pure, Re, bc_index, pressure=False, hard_bc=False):

    device = model_out_pure.device

    # find where boundaries end
    min_list = []
    for patch in bc_index['Boundary Indices']:
        min_list += [bc_index['Boundary Indices'][patch].min()]
    min_bc_index = np.min(min_list)

    # create objects (internal field only)
    B = model_out_pure.shape[0]
    C = model_out_pure.shape[-1]
    resolution = int(np.sqrt(min_bc_index))
    mesh_coords = model_input_coords[0,:min_bc_index,:].reshape(resolution, resolution, 2)
    u = torch.zeros(B,resolution+2, resolution+2,C).to(device)
    dx = (mesh_coords[0,1,0] - mesh_coords[0,0,0]).item()
    dy = (mesh_coords[1,0,1] - mesh_coords[0,0,1]).item()
    
    if hard_bc: 
        raise NotImplementedError('Hard Boundaries not available yet (distance from cell to wall is not dx)')
        u[:,1:-1,1:-1, :] = model_out_pure[:,:min_bc_index,:].reshape(B,resolution,resolution,C)
        u[:,-1  , :  , :] = model_out_pure[:,lid_indices,:]
        u[:, :  , 0  , :] = model_out_pure[:,left_indices,:]
        u[:, 0  , :  , :] = model_out_pure[:,bottom,:]
        u[:, :  ,-1  , :] = model_out_pure[:,right_iniec,:]
    else:
        u[:,1:-1,1:-1, :] = model_out_pure[:,:min_bc_index,:].reshape(B,resolution,resolution,C)
       
        # Lid
        u[:,-1  , :  , 0] = -u[:,-2  , :  , 0] + 2.0
        u[:,-1  , :  , 1] = -u[:,-2  , :  , 1] 
        u[:,-1  , :  , 2] =  u[:,-2  , :  , 2]

        # Left Wall
        u[:, :  , 0  , :2] = -u[:, :  , 1  , :2]
        u[:, :  , 0  , 2]  =  u[:, :  , 1  , 2]

        # Bottom Wall
        u[:, 0  , :  , :2] = -u[:, 0  , :  , :2]
        u[:, 0  , :  , 2] =  u[:, 0  , :  , 2]

        # Right Wall
        u[:, :  ,-1  , :2] = -u[:, :  ,-1  , :2]
        u[:, :  ,-1  , 2] = -u[:, :  ,-1  , 2]

    # Stack and Repeat Re for tensor multiplication
    Re = Re.squeeze(-1)

    # gradients in internal zone
    u_y  = (u[:, 2:  , 1:-1, 0] -   u[:,  :-2, 1:-1, 0]) / (2*dy)
    u_x  = (u[:, 1:-1, 2:  , 0] -   u[:, 1:-1,  :-2, 0]) / (2*dx)
    u_yy = (u[:, 2:  , 1:-1, 0] - 2*u[:, 1:-1, 1:-1, 0] + u[:,  :-2, 1:-1, 0]) / (dy**2)
    u_xx = (u[:, 1:-1, 2:  , 0] - 2*u[:, 1:-1, 1:-1, 0] + u[:, 1:-1,  :-2, 0]) / (dx**2)

    v_y  = (u[:, 2:  , 1:-1, 1] -   u[:,  :-2, 1:-1, 1]) / (2*dy)
    v_x  = (u[:, 1:-1, 2:  , 1] -   u[:, 1:-1,  :-2, 1]) / (2*dx)
    v_yy = (u[:, 2:  , 1:-1, 1] - 2*u[:, 1:-1, 1:-1, 1] + u[:,  :-2, 1:-1, 1]) / (dy**2)
    v_xx = (u[:, 1:-1, 2:  , 1] - 2*u[:, 1:-1, 1:-1, 1] + u[:, 1:-1,  :-2, 1]) / (dx**2)

    p_y  = (u[:, 2:  , 1:-1, 2] - u[:,  :-2, 1:-1, 2]) / (2*dy)
    p_x  = (u[:, 1:-1, 2:  , 2] - u[:, 1:-1,  :-2, 2]) / (2*dx)

    # No time derivative as we are assuming steady state solution
    f0 = (u_x + v_y)
    f1 = u[:,1:-1,1:-1, 0]*u_x + u[:,1:-1,1:-1, 1]*u_y - (1/Re) * (u_xx + u_yy) + p_x
    f2 = u[:,1:-1,1:-1, 0]*v_x + u[:,1:-1,1:-1, 1]*v_y - (1/Re) * (v_xx + v_yy) + p_y

    derivatives = {
                   'u_x':u_x, 'u_y':u_y, 'v_x':v_x, 'v_y':v_y, 'p_x':p_x, 'p_y':p_y,
                   'u_xx':u_xx, 'u_yy':u_yy, 'v_xx':v_xx, 'v_yy':v_yy
                   }

    # Pressure correction (incomressible condition)
    if pressure:
        f3 = (u[...,0]**2 + u[...,1]**2)*1/2 - u[...,2]
        return [f0,f1,f2,f3], derivatives
    else:
        return [f0,f1,f2], derivatives

def bc_numerical(model_input_coords, model_out_pure, bc_index, loss_function=torch.nn.MSELoss()):
    device = model_out_pure.device
    
    # find where boundaries end
    min_list = []
    for patch in bc_index['Boundary Indices']:
        min_list += [bc_index['Boundary Indices'][patch].min()]
    min_bc_index = np.min(min_list)

    # create objects (internal field only)
    B = model_out_pure.shape[0]
    C = model_out_pure.shape[-1]
    resolution = int(np.sqrt(min_bc_index))
    #mesh_coords = model_input_coords[0,:min_bc_index,:].reshape(resolution, resolution, 2)
    u = model_out_pure[:,:min_bc_index,:].reshape(B,resolution,resolution,C).to(device)

    d_loss = []
    d_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Lid'].flatten(),0], torch.ones(B,resolution, dtype=float, device=device))]
    d_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Lid'].flatten(),1], torch.zeros(B,resolution, dtype=float, device=device))]
    d_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Left Wall'].flatten(),:2], torch.zeros(B,resolution,2, dtype=float, device=device))]
    d_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Bottom Wall'].flatten(),:2], torch.zeros(B,resolution,2, dtype=float, device=device))]
    d_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Right Wall'].flatten(),:2], torch.zeros(B,resolution,2, dtype=float, device=device))]
    d_loss = torch.mean(torch.stack(d_loss))

    vn_loss = []
    vn_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Lid'].flatten(),2], u[:,-1,:,2])]
    vn_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Left Wall'].flatten(),2], u[:,:,0,2])]
    vn_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Bottom Wall'].flatten(),2], u[:,0,:,2])]
    vn_loss += [loss_function(model_out_pure[:,bc_index['Boundary Indices']['Right Wall'].flatten(),2], u[:,:,-1,2])]
    vn_loss = torch.mean(torch.stack(vn_loss))

    return d_loss, vn_loss 

# Loss Function application and construction function
def ns_pde_numerical_loss(model_input_coords, model_out_pure, Re, bc_index, loss_function=torch.nn.MSELoss(), pressure=False):

    pde_eqns, derivatives = ns_pde_numerical(model_input_coords, model_out_pure, Re, bc_index, pressure=pressure, hard_bc=False)

    loss_list = list()
    for pde_eqn in pde_eqns:
        pde_loss = loss_function(pde_eqn,torch.zeros_like(pde_eqn))
        loss_list.append(pde_loss.float())

    d_loss, vn_loss = bc_numerical(model_input_coords, model_out_pure, bc_index)
    
    bc_loss_list = list()
    bc_loss_list.append(d_loss.float())
    bc_loss_list.append(vn_loss.float())

    print(loss_list)
    print(bc_loss_list)
    return loss_list, bc_loss_list, derivatives