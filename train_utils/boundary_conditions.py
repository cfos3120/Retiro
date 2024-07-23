
def bc_loss(model_y,y,bc_index,derivatives, loss_function):

    # Dirichlet Boundary Conditions
    # Easiest way is to just compare to existing solution at boundary
    # upperWall, lowerWall, inlet, outlet

    d_loss = 0
    for patch in bc_index:
        d_loss += loss_function(model_y[:,bc_index[patch],:],y[:,bc_index[patch],:])

    # Von Neumann Boundary Conditions 
    # Currently for Step Case only
    # TODO create a mapping function between openfoam initial case

    # outlet Velocity Conditions are zero Gradient
    # inlet, upperWall, lowerWall Pressure Conditions are zero Gradient
    vn_loss = 0
    vn_loss += loss_function(derivatives['u_x'][:,bc_index['outlet']])
    vn_loss += loss_function(derivatives['u_y'][:,bc_index['outlet']])
    
    for patch in ['inlet', 'upperWall', 'lowerWall']:
        vn_loss += loss_function(derivatives['p_x'][:,bc_index[patch]])
        vn_loss += loss_function(derivatives['p_y'][:,bc_index[patch]])

    loss_list = list()
    loss_list.append(d_loss)
    loss_list.append(vn_loss)
    return loss_list
