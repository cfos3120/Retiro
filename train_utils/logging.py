import numpy as np
import os
import torch

class total_loss_list():
    def __init__(self, model_config=None, training_config=None, data_config=None):
        super(total_loss_list, self).__init__()
        self.dictionary = dict()

        # Create dictionary object with model class data
        if model_config is not None:
            self.dictionary['Model Configuration'] = model_config

        # Create dictionary object with training settings
        if model_config is not None:
            self.dictionary['Training Configuration'] = training_config

        # Create dictionary object with data settings
        if model_config is not None:
            self.dictionary['Data Configuration'] = data_config
        
    def update(self, loss_dict):
        for key_name in loss_dict.keys():
            if key_name not in self.dictionary.keys():
                self.dictionary[key_name] = []
            self.dictionary[key_name].append(loss_dict[key_name])
    
    def fetch_dict(self):
        return self.dictionary
    
class loss_aggregator():
    def __init__(self):
        super(loss_aggregator, self).__init__()
        self.main_loss_dict = total_loss_list()
        self.aggregated_dict = {}

    def add(self, loss_dict):
        self.main_loss_dict.update(loss_dict)

    def aggregate(self):
        for key in self.main_loss_dict.fetch_dict().keys():
            self.aggregated_dict[key] = np.mean(self.main_loss_dict.fetch_dict()[key])

        return self.aggregated_dict

def save_checkpoint(path, name, model=None, loss_dict=None, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    if model != None:
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        if optimizer is not None:
            optim_dict = optimizer.state_dict()
        else:
            optim_dict = 0.0

        torch.save({
            'model': model_state_dict,
            'optim': optim_dict
        }, ckpt_dir + name + '.pt')
        print('Checkpoint is saved at %s' % ckpt_dir + name + '.pt')

    if loss_dict != None:
        np.save(ckpt_dir + name + '_results', loss_dict)
        print("Training Dictionary Saved in Same Location")