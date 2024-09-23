""" Script to train variational inference network for posterior of subspace parameters """
import os
import argparse
import pytorch_lightning as pl

# My imports

import gc
from chemprop_utils.train_utils import train_vi_one_epoch
import subspace_inference.utils as subspace_utils
from subspace_inference.proj_model import SubspaceModel
from subspace_inference.vi_model import VIModel
import math
import torch
import numpy as np


import chemprop
from chemprop.data import get_data, get_task_names, MoleculeDataLoader
from chemprop.utils import create_logger, load_checkpoint
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
from chemprop.train.loss_functions import get_loss_func


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prop_name', default = 'DRD2',type=str ,help='property name')
    parser.add_argument('--AS_dim', default = '10',type=int ,help='dimension of active subspace')
    cli_args = parser.parse_args()


    arguments = [
        '--data_path', f'data/{cli_args.prop_name.lower()}/train_split.csv',
        # '--separate_val_path', 'data/validation_split.csv',
        '--separate_test_path', f'data/{cli_args.prop_name.lower()}/test_split.csv',
        '--dataset_type', 'classification',
        '--save_dir', f'log/{cli_args.prop_name.lower()}/basic',
        '--checkpoint_path', f'{cli_args.prop_name.lower()}_checkpoints/basic/model_0/model.pt',
        '--metric', 'precision',
        #'--extra_metrics', 'recall,precision,accuracy',
        '--quiet',
        '--epochs', '10',
        '--batch_size', '64',
        '--num_workers', '2',
        # '--features_generator', 'rdkit_2d_normalized', '--no_features_scaling',
        '--seed', '0',
    ]

    if cli_args.prop_name == 'DRD2':
        arguments += ['--separate_val_path', f'data/{cli_args.prop_name.lower()}/validation_split.csv']


    args = chemprop.args.TrainArgs().parse_args(arguments)
    args.extra_metrics= ['auc','recall','accuracy']
    print(args.metrics)
    args.task_names = get_task_names(
            path=args.data_path,
            smiles_columns=args.smiles_columns,
            target_columns=args.target_columns,
            ignore_columns=args.ignore_columns,
            loss_function=args.loss_function,
        )
    #args.metrics = ['auc', 'prc-auc','recall', 'precision','accuracy']

    logger = create_logger(name='log_file', save_dir=args.save_dir, quiet=args.quiet)

    train_data = get_data(
            path=args.data_path,
            args=args,
            logger=logger,
            skip_none_targets=True,
            data_weights_path=args.data_weights_path
        )
    args.features_size = train_data.features_size()

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = data.atom_descriptors_size()
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = data.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
    if args.bond_descriptors == 'descriptor':
        args.bond_descriptors_size = data.bond_descriptors_size()
    elif args.bond_descriptors == 'feature':
        args.bond_features_size = data.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)

    train_dataloader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed)

    print(len(train_dataloader.dataset))

    pretrained_model = load_checkpoint(args.checkpoint_path, device=args.device)

    all_params = [p[0] for p in pretrained_model.named_parameters()]
    all_params_count = [p[1].numel() for p in pretrained_model.named_parameters()]
    all_params_count_with_grad = [p[1].numel() for p in pretrained_model.named_parameters() if p[1].requires_grad]

    # Parse arguments
    pl.seed_everything(args.seed)
    subspace_method = 'AS'
    
    result_dir = os.path.join(args.save_dir,subspace_method)
    # os.makedirs(result_dir, exist_ok = True)
    
    subunit = [p[0] for p in pretrained_model.named_parameters() if p[1].requires_grad]
    
    
    Np = sum([p[1].numel() for p in pretrained_model.named_parameters() if p[0] in subunit])

    print(f'Total number of parameters: {sum(all_params_count)}')
    print(f'Total number of parameters requiring grad: {Np}')
    



    active_dim = cli_args.AS_dim #20    
    
    # load subspace data

    subspace_data = np.load(os.path.join(result_dir,f'singular_values_{args.seed}_AS_dim_{active_dim}.npz'), allow_pickle = True)
    mean = torch.cat([p[1].data.flatten() for p in pretrained_model.named_parameters() if p[0] in subunit]).cpu().detach()
    # load the AS's mean
    subspace_obj = SubspaceModel(mean, torch.FloatTensor(subspace_data['V'][:active_dim,:]))


    assert subspace_obj.rank==active_dim, f"subspace has {subspace_obj.rank} dimension instead of active_dim"
    
    # train VI model
    init_sigma = 1e-3
    prior_sigma = 5.
    
    temperature = 1.

    
    for p in pretrained_model.named_parameters():
        if p[0] not in subunit:
            p[1].requires_grad_(False)


    
    subunit_without_last = ['.'.join(p[0].split('.')[:-1]) for p in pretrained_model.named_parameters() if p[0] in subunit]
    print(subunit_without_last)


    vi_model = VIModel(
        subspace=subspace_obj,
        init_inv_softplus_sigma=math.log(math.exp(init_sigma) - 1.0),
        prior_log_sigma=math.log(prior_sigma),
        base=pretrained_model,
        selected_params_list = subunit_without_last,
    )
    use_gpu = True
    if use_gpu:
        vi_model = vi_model.cuda()
    # gc.collect()

    # n_samples = len(train_dataloader.dataset)*hparams.batch_size # this was used in kl loss of vi
    optimizer = torch.optim.Adam(vi_model.parameters(), lr=0.001)
    vi_model.train()
    loss_stats_to_save = {'total_loss':[],'base_loss':[], 'vi_kl_loss': []}
    print('getting ready to train')
    vi_model.base_model.beta = 0.005    
    for epoch in range(args.epochs):
        #print(epoch)
        # if epoch == 500:
        #     # vi_model.base_model.beta = 0.005*epoch/150
        #     vi_model.base_model.beta = 0.005       
        loss_stats = train_vi_one_epoch( vi_model, train_dataloader,get_loss_func(args),optimizer,temperature,args)
        
        loss_stats_to_save['total_loss'].append(loss_stats['total_loss'])
        loss_stats_to_save['base_loss'].append(loss_stats['base_loss'])
        # loss_stats_to_save['base_rec_loss'].append(loss_stats['base_rec_loss'])
        # loss_stats_to_save['base_kl_loss'].append(loss_stats['base_kl_loss'])
        loss_stats_to_save['vi_kl_loss'].append(loss_stats['vi_kl_loss'])

        if epoch % 1 == 0:
            print(epoch, f" chemprop loss : {loss_stats_to_save['base_loss'][-1]}, VI_KL: {loss_stats_to_save['vi_kl_loss'][-1]},"+
                    f" Total loss: {loss_stats_to_save['total_loss'][-1]}")
        # if epoch == 700:
        #     subspace_utils.adjust_learning_rate(optimizer, 0.0005)
        
    np.savez_compressed(os.path.join(result_dir, f'vi_params_{args.seed}_AS_dim_{active_dim}.npz'), mu = vi_model.mu.cpu().detach().numpy(), sigma = torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu()).numpy())
    np.savez_compressed(os.path.join(result_dir,f'vi_training_{args.seed}_AS_dim_{active_dim}.npz'),**loss_stats_to_save)
    torch.save(vi_model.state_dict(), os.path.join(result_dir,f'vi_model_{args.seed}_AS_dim_{active_dim}.pt'))
