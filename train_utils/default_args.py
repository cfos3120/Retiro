
# Fetch default args
def get_default_args():
    
    dataset_args = dict()
    dataset_args['name']        = 'Cavity'
    dataset_args['L']           = 1.0
    dataset_args['sub_x']       = 8
    dataset_args['normalize_y'] = True
    dataset_args['normalize_x'] = True
    dataset_args['normalize_f'] = True
    dataset_args['vertex']      = True
    dataset_args['boundaries']  = True
    dataset_args['inference']   = True
    dataset_args['train_ratio'] = 0.7
    dataset_args['seed']        = 42
    dataset_args['batchsize']   = 4
    dataset_args['random_coords'] = True
    dataset_args['train'] = True
    dataset_args['keys only resolution'] = 32
    dataset_args['key_range_min'] = 1.0
    dataset_args['key_range_max'] = 100.0
    dataset_args['bc_input_f']    = False # For Step

    model_args = dict()
    model_args['trunk_size']        = 2
    model_args['theta_size']        = 0
    model_args['branch_sizes']      = [1]
    model_args['output_size']       = 3
    model_args['n_layers']          = 3
    model_args['n_hidden']          = 64  
    model_args['n_head']            = 1
    model_args['attn_type']         = 'linear'
    model_args['ffn_dropout']       = 0.0
    model_args['attn_dropout']      = 0.0
    model_args['mlp_layers']        = 2
    model_args['act']               = 'gelu'
    model_args['hfourier_dim']      = 0
    model_args['gating']            = True
    model_args['init_w']            = False
    
    training_args = dict()
    training_args['epochs']                 = 1
    training_args['base_lr']                = 0.001
    training_args['weight-decay']           = 0.00005
    training_args['grad-clip']              = 1000.0    
    training_args["save_dir"]               = 'retiro'
    training_args["save_name"]              = 'test'
    training_args["scheduler"]              = 'Step'
    training_args['Hybrid_type']            = 'Monitor'
    training_args['Key_only_batches']       = 0
    training_args['Secondary_optimizer']    = False
    training_args['eval_while_training']    = True
    training_args['dynamic_balance']        = False

    return dataset_args, model_args, training_args


# Make adjustments based on ArgParser inputs
def args_override(dataset_args, model_args, training_args, argparser_args):
    dataset_args['sub_x']           = argparser_args.sub_x
    dataset_args['batchsize']       = argparser_args.batch_size
    dataset_args['file_path']       = argparser_args.path
    dataset_args['random_coords']   = argparser_args.rand_cood == 1
    dataset_args['normalize_f']     = argparser_args.normalize_f == 1
    dataset_args['inference']       = argparser_args.inference == 1
    dataset_args['train_ratio']     = argparser_args.datasplit
    dataset_args['keys only resolution'] = argparser_args.ko_res
    dataset_args['name']            = argparser_args.data_name

    training_args['DP']             = argparser_args.DP == 1
    training_args['Hybrid_type']    = argparser_args.Hybrid_type
    training_args['base_lr']        = argparser_args.lr
    training_args['step_size']      = argparser_args.step_size
    training_args["save_name"]      = argparser_args.name
    training_args["save_dir"]       = argparser_args.dir
    training_args['epochs']         = argparser_args.epochs
    training_args['scheduler']      = argparser_args.scheduler
    training_args['ckpt']           = argparser_args.ckpt_path
    training_args['Key_only_batches'] = argparser_args.Key_only_batches
    training_args['Secondary_optimizer'] = argparser_args.Secondary_optimizer == 1
    training_args['dynamic_balance'] = argparser_args.dynamic_balance == 1

    model_args['init_w']            = argparser_args.init_w == 1
    model_args['gating']            = argparser_args.gating == 1
    model_args['n_layers']          = argparser_args.layers
    model_args['n_hidden']          = argparser_args.n_hidden  

    return dataset_args, model_args, training_args
