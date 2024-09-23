'''
TODO:
    load the SGD_PCA's mean
'''


from tqdm import tqdm
import os
import argparse
import pytorch_lightning as pl

# My imports

import gc
from chemprop_utils.train_utils import eval_model
import subspace_inference.utils as subspace_utils
from subspace_inference.proj_model import SubspaceModel
from subspace_inference.vi_model import VIModel
import math
import torch
import numpy as np


def load_weights(model, w,param_list = None):
    offset = 0
    for m in model.named_parameters():
        if m[0] in param_list or param_list is None:
            size = m[1].numel()
            m[1].data = w[offset:offset+size].reshape_as(m[1].data)
            offset += size


class VIModel_sampler(torch.nn.Module):
    def __init__(self, subspace, init_inv_softplus_sigma=-3.0, 
                eps=1e-6, with_mu=True, *args, **kwargs):
        super(VIModel_sampler, self).__init__()

        self.subspace = subspace
        self.rank = self.subspace.rank

        #self.prior_log_sigma = prior_log_sigma # not used for sampling
        self.eps = eps

        self.with_mu = with_mu
        if with_mu:
            self.mu = torch.nn.Parameter(torch.zeros(self.rank))
        self.inv_softplus_sigma = torch.nn.Parameter(torch.empty(self.rank).fill_(init_inv_softplus_sigma))


    def sample(self, scale=1.):
        device = self.inv_softplus_sigma.device
        sigma = torch.nn.functional.softplus(self.inv_softplus_sigma.detach()) + self.eps
        if self.with_mu:
            z = self.mu + torch.randn(self.rank, device=device) * sigma * scale
        else:
            z = torch.randn(self.rank, device=device) * sigma * scale
        w = self.subspace(z)
        return w


import chemprop
from chemprop.data import get_data, get_task_names, MoleculeDataLoader
from chemprop.utils import create_logger, load_checkpoint
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
from chemprop.train.loss_functions import get_loss_func

from sklearn.metrics import roc_auc_score, precision_score, recall_score

def print_metric(test_Y, pred_Y):
    roc_auc = roc_auc_score(test_Y, pred_Y, average='weighted')
    print(roc_auc)
    print(f"test precision: {precision_score(test_Y, pred_Y>0.5)}")
    print(f"test recall: {recall_score(test_Y, pred_Y>0.5)}")

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', default = 0,type=int ,help='trial seed')
    parser.add_argument('--prop_name', default = 'DRD2',type=str ,help='property name')
    parser.add_argument('--AS_dim', default = '10',type=int ,help='dimension of active subspace')
    parser.add_argument('--num_models', default = '10',type=int ,help='number of model samples in Bayesian inference')

    cli_args = parser.parse_args()

    ckpt_map = {'DRD2': 'drd2_checkpoints/basic/model_0/model.pt',
                'GSK3B': 'gsk3b_checkpoints/basic/model_0/model.pt',
                'JNK3': 'jnk3_checkpoints/basic/model_0/model.pt'}


    arguments = [
        '--data_path', f'data/{cli_args.prop_name.lower()}/train_split.csv',
        # '--separate_val_path', 'data/validation_split.csv',
        '--separate_test_path', f'data/{cli_args.prop_name}_screening_cands.csv', # change this test_split.csv with candidate smiles and dummy label
        '--dataset_type', 'classification',
        '--save_dir', f'log/{cli_args.prop_name.lower()}/basic',
        '--checkpoint_path', ckpt_map[cli_args.prop_name],
        '--metric', 'precision',
        #'--extra_metrics', 'recall,precision,accuracy',
        '--quiet',
        '--epochs', '5',
        '--batch_size', '64',
        '--num_workers', '2',
        # '--features_generator', 'rdkit_2d_normalized', '--no_features_scaling',
        '--seed', '0',
    ]

    if cli_args.prop_name == 'DRD2':
        arguments += ['--separate_val_path', f'data/{cli_args.prop_name.lower()}/validation_split.csv']


    args = chemprop.args.TrainArgs().parse_args(arguments)

    args.task_names = get_task_names(
            path=args.data_path,
            smiles_columns=args.smiles_columns,
            target_columns=args.target_columns,
            ignore_columns=args.ignore_columns,
            loss_function=args.loss_function,
        )
    #args.metrics = ['auc', 'prc-auc','recall', 'precision','accuracy']

    

    test_data = get_data(
            path=args.separate_test_path,
            args=args,
            logger=None,
            skip_none_targets=True,
            # data_weights_path=args.data_weights_path
        )
    args.features_size = test_data.features_size()

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

    test_dataloader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=0,
        # class_balance=args.class_balance,
        shuffle=False,
        # seed=args.seed,
        )

    print(len(test_dataloader.dataset))

    pretrained_model = load_checkpoint(args.checkpoint_path, device=args.device)

    all_params = [p[0] for p in pretrained_model.named_parameters()]
    all_params_count = [p[1].numel() for p in pretrained_model.named_parameters()]
    all_params_count_with_grad = [p[1].numel() for p in pretrained_model.named_parameters() if p[1].requires_grad]

    # Parse arguments
    pl.seed_everything(cli_args.trial)
    subspace_method = 'AS'
    
    result_dir = os.path.join(args.save_dir,subspace_method)
    # os.makedirs(result_dir, exist_ok = True)
    
    subunit = [p[0] for p in pretrained_model.named_parameters() if p[1].requires_grad]
    
    
    Np = sum([p[1].numel() for p in pretrained_model.named_parameters() if p[0] in subunit])

    print(f'Total number of parameters: {sum(all_params_count)}')
    print(f'Total number of parameters requiring grad: {Np}')


    print('Inference using pre-trained')
    
    _ , all_preds = eval_model(pretrained_model, test_dataloader, get_loss_func(args) ,args, use_gpu = True)

    print(len(all_preds))
    print(len(test_dataloader.dataset.targets()))
    assert len(all_preds) == len(test_dataloader.dataset.targets())
    # sigmoid_func = lambda x: 1/(1+np.exp(-x))
    
    PTM_prediction = np.array(all_preds)
    
    active_dim = cli_args.AS_dim
    # load subspace data

    subspace_data = np.load(os.path.join(result_dir,f'singular_values_{args.seed}_AS_dim_{active_dim}.npz'), allow_pickle = True)
    mean = torch.cat([p[1].data.flatten() for p in pretrained_model.named_parameters() if p[0] in subunit]).cpu().detach()
    # load the AS's mean
    subspace_obj = SubspaceModel(mean, torch.FloatTensor(subspace_data['V'][:active_dim,:]))


    assert subspace_obj.rank==active_dim, f"subspace has {subspace_obj.rank} dimension instead of active_dim"

    # we have vi_params_obj which we will initialize with learned parameters, then we sample from it
    vi_model_sampler = VIModel_sampler(subspace=subspace_obj)
    vi_model_PATH = os.path.join(result_dir,f'vi_model_{args.seed}_AS_dim_{active_dim}.pt')
    vi_model_sampler.load_state_dict(torch.load(vi_model_PATH), strict=False)


    num_model_samples = cli_args.num_models # for cifar it was 30.
    # sample_per_model = int(num_queries_to_do/num_model_samples)
    print(vi_model_sampler.mu)
    print(vi_model_sampler.inv_softplus_sigma)

    print('Bayesian model sampling')

    all_preds_list = []
    for idx in tqdm(range(num_model_samples)):

        w = vi_model_sampler.sample()

        # set the specific weights
        load_weights(pretrained_model, w ,param_list = subunit)
        # do the evaluation run
        
        _ , all_preds = eval_model(pretrained_model, test_dataloader, get_loss_func(args) ,args, use_gpu = True)
        all_preds_list.append(all_preds)
        

    BMA_preds = np.hstack(all_preds_list).mean(axis = 1)
    BMA_unc = np.hstack(all_preds_list).std(axis = 1)

    cands_list = [ x.smiles[0] for x in test_data]
    
    np.savez_compressed(os.path.join(result_dir,
        f'{cli_args.prop_name}_screening_trial_{cli_args.trial}_AS_dim_{active_dim}_num_models_{num_model_samples}.npz'),
        BMA_preds=BMA_preds, BMA_unc = BMA_unc,
        PTM_pred=PTM_prediction, cands_list = cands_list)
    
    
