import torch

# Autograd Calculation of Gradients and Navier-Stokes Equations
def ns_pde_autograd(model_input_coords, model_out, Re):

    # Stack and Repeat Re for tensor multiplication
    Re = Re.squeeze(-1)

    u = model_out[..., 0]
    v = model_out[..., 1]
    p = model_out[..., 2]

    # First Derivatives
    u_out = torch.autograd.grad(u.sum(), model_input_coords, create_graph=True)[0]
    v_out = torch.autograd.grad(v.sum(), model_input_coords, create_graph=True)[0]
    p_out = torch.autograd.grad(p.sum(), model_input_coords, create_graph=True)[0]

    u_x = u_out[..., 0]
    u_y = u_out[..., 1]

    v_x = v_out[..., 0]
    v_y = v_out[..., 1]

    p_x = p_out[..., 0]
    p_y = p_out[..., 1]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x.sum(), model_input_coords, create_graph=True)[0][..., 0]
    u_yy = torch.autograd.grad(u_y.sum(), model_input_coords, create_graph=True)[0][..., 1]
    v_xx = torch.autograd.grad(v_x.sum(), model_input_coords, create_graph=True)[0][..., 0]
    v_yy = torch.autograd.grad(v_y.sum(), model_input_coords, create_graph=True)[0][..., 1]

    # Continuity equation
    f0 = u_x + v_y

    # Navier-Stokes equation
    f1 = u*u_x + v*u_y - (1/Re) * (u_xx + u_yy) + p_x
    f2 = u*v_x + v*v_y - (1/Re) * (v_xx + v_yy) + p_y

    derivatives = {
                   'u_x':u_x, 'u_y':u_y, 'v_x':v_x, 'v_y':v_y, 'p_x':p_x, 'p_y':p_y,
                   'u_xx':u_xx, 'u_yy':u_yy, 'v_xx':v_xx, 'v_yy':v_yy
                   }

    return [f0,f1,f2], derivatives


# Loss Function application and construction function
def ns_pde_autograd_loss(model_input_coords, model_out, Re, loss_function=torch.nn.MSELoss()):

    pde_eqns,__ = ns_pde_autograd(model_input_coords, model_out, Re)
    
    loss_list = list()
    for pde_eqn in pde_eqns:
        pde_loss = loss_function(pde_eqn,torch.zeros_like(pde_eqn))
        loss_list.append(pde_loss)

    return loss_list

# For Autograd to work well we need to wrap the model to include input normalization
class wrapped_model(torch.nn.Module):

    '''
    TODO: Saving and loading checkpoints using a wrapped model may be malaligned when
    using a non-wrapped model and vice versa. 
    '''

    def __init__(self, model, query_normalizer=None, output_normalizer=None):
        super(wrapped_model, self).__init__()
        self.query_normalizer = query_normalizer
        self.output_normalizer = output_normalizer
        self.model = model

    def forward(self,x,inputs):
        if self.query_normalizer is not None:
            x = self.query_normalizer.transform(x, inverse=False)
        
        out = self.model(x,inputs=inputs)

        if self.output_normalizer is not None:
            out = self.output_normalizer.transform(out, inverse=True)

        return out


