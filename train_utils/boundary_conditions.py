from data_utils.utils import parse_arguments
import numpy as np
import torch

#ARGS = parse_arguments()

boundary_mapping = {'Cavity' :{
                        'D_BC' : {
                            'p' : [],
                            'u' : ['movingWall','fixedWalls'],
                            'v' : ['movingWall','fixedWalls']
                            },
                        'VN_BC' : {
                            'p' : ['movingWall','fixedWalls'],
                            'u' : [],
                            'v' : []
                            }
                        },
                    
                    'Step' :{
                        'D_BC' : {
                            'p' : ['outlet'],
                            'u' : ['inlet','upperWall','lowerWall'],
                            'v' : ['inlet','upperWall','lowerWall']
                            },
                        'VN_BC' : {
                            'p' : ['inlet', 'upperWall' ,'lowerWall'],
                            'u' : ['outlet'],
                            'v' : ['outlet']
                            }
                        }
                    }



def bc_loss(model_y,y,bc_index,derivatives, loss_function, ARGS):

    # Dirichlet Boundary Conditions
    # Easiest way is to just compare to existing solution at boundary
    # upperWall, lowerWall, inlet, outlet
    
    d_loss = []
    for patch in boundary_mapping[ARGS.name]['D_BC']['u']:
        d_loss += [loss_function(model_y[:,bc_index[patch],0],y[:,bc_index[patch],0])]
    for patch in boundary_mapping[ARGS.name]['D_BC']['v']:
        d_loss += [loss_function(model_y[:,bc_index[patch],1],y[:,bc_index[patch],1])]
    for patch in boundary_mapping[ARGS.name]['D_BC']['p']:
        d_loss += [loss_function(model_y[:,bc_index[patch],2],y[:,bc_index[patch],2])]
    d_loss = torch.mean(torch.stack(d_loss))

    # d_loss = 0
    # for patch in bc_index:
    #     print(f'Patch {patch}: solution {model_y[:,bc_index[patch],:]}')
    #     d_loss += loss_function(model_y[:,bc_index[patch],:],y[:,bc_index[patch],:])

    # Von Neumann Boundary Conditions 
    # Currently for Step Case only
    # TODO create a mapping function between openfoam initial case

    # outlet Velocity Conditions are zero Gradient
    # inlet, upperWall, lowerWall Pressure Conditions are zero Gradient
    # vn_loss = 0
    # vn_loss += loss_function(derivatives['u_x'][:,bc_index['outlet']])
    # vn_loss += loss_function(derivatives['u_y'][:,bc_index['outlet']])
    
    vn_loss = []
    for patch in boundary_mapping[ARGS.name]['VN_BC']['u']:
        vn_loss += [loss_function(derivatives['u_x'][:,bc_index[patch]])]
        vn_loss += [loss_function(derivatives['u_y'][:,bc_index[patch]])]
    for patch in boundary_mapping[ARGS.name]['VN_BC']['v']:
        vn_loss += [loss_function(derivatives['v_x'][:,bc_index[patch]])]
        vn_loss += [loss_function(derivatives['v_y'][:,bc_index[patch]])]
    for patch in boundary_mapping[ARGS.name]['VN_BC']['p']:
        vn_loss += [loss_function(derivatives['p_x'][:,bc_index[patch]])]
        vn_loss += [loss_function(derivatives['p_y'][:,bc_index[patch]])]
    vn_loss = torch.mean(torch.stack(vn_loss))

    # for patch in ['inlet', 'upperWall', 'lowerWall']:
    #     vn_loss += loss_function(derivatives['p_x'][:,bc_index[patch]])
    #     vn_loss += loss_function(derivatives['p_y'][:,bc_index[patch]])

    loss_list = list()
    loss_list.append(d_loss)
    loss_list.append(vn_loss)
    return loss_list
