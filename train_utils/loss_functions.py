import torch

class LP_custom(object):
    # Loss function is based on the DGL weighted loss function (only it is dgl package free)
    
    def __init__(self):
        super(LP_custom, self).__init__()

    def avg_pool(self,input):
        #r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
        batch_size = input.shape[0] # shape: Batch, Nodes, Channels
        num_nodes = input.shape[1]
        pooled_value = (1/num_nodes)*torch.sum(input,dim=1)

        return pooled_value
    
    def __call__(self, x, y=None):

        if y is not None:
            losses = (self.avg_pool(((x - y).abs() ** 2)) + 1e-8) ** (1 / 2)
        else:
            losses = (self.avg_pool((x.abs() ** 2)) + 1e-8) ** (1 / 2)
        loss = losses.mean()
        return loss