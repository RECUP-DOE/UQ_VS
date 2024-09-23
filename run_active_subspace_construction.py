""" Script t construct active subspace either around pretrained weights or SGD-PCA mean """
import os
import argparse
import pytorch_lightning as pl

# My imports
from sklearn.utils.extmath import randomized_svd

from chemprop_utils.train_utils import one_gradient_sample
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

def init_weights(model, pretrained_model, sig_nn, param_list = None, Np = None):
    
    if Np is not None:
        u = torch.randn(1,Np)
        u /= torch.norm(u, dim=-1, keepdim=True)
        #u *= sig_nn*(1-torch.rand(1))
        u *= sig_nn
        offset = 0
    
    for m,m_pretrained in zip(model.named_parameters(), pretrained_model.named_parameters()):
        # m = name_modules[1]
        # m_pretrained = name_modules_pretrained[1]
        # m_name = name_modules[0]
        if m[0] in param_list or param_list is None:
            if Np is None:
                m[1].data = torch.normal(mean=m_pretrained[1].data, std=sig_nn)
            else:
                size = m_pretrained[1].numel()
                u_sub = u[0,offset:offset+size]
                m[1].data = m_pretrained[1].data + u_sub.reshape_as(m_pretrained[1].data)
                offset += size
            # if isinstance(m[1], torch.nn.Linear):
            #     m[1].weight.data = torch.normal(mean=m_pretrained[1].weight.data, std=sig_nn)
            #     if m[1].bias is not None:
            #         m[1].bias.data = torch.normal(mean=m_pretrained[1].bias.data, std=sig_nn)
            # else:
            #     raise NotImplementedError(f'expecting torch.nn.Linear; but got {m}')
        # else:
        #     m[1].data = m_pretrained[1].data
                    


# def one_gradient_sample(model,input_sample, optimizer, use_gpu = False):
#     model.train()

#     if use_gpu:
#         input = input_sample.cuda(non_blocking=True)
#     else:
#         input = input_sample

#     # get loss
#     loss = model(input)
    
#     # print(loss)

#     optimizer.zero_grad()
#     loss.backward()
#     # optimizer.step()



    
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
    #    '--separate_val_path', 'data/jnk3/validation_split.csv',
        '--separate_test_path', f'data/{cli_args.prop_name.lower()}/test_split.csv',
        '--dataset_type', 'classification',
        '--save_dir', f'log/{cli_args.prop_name.lower()}/basic',
        '--checkpoint_path', f'{cli_args.prop_name.lower()}_checkpoints/basic/model_0/model.pt',
        '--metric', 'precision',
        #'--extra_metrics', 'recall,precision,accuracy',
        '--quiet',
        '--epochs', '20',
        '--batch_size', '64',
        '--num_workers', '2',
    #    '--features_generator', 'rdkit_2d_normalized', '--no_features_scaling',
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
        num_workers=0,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    # val_data_loader = MoleculeDataLoader(
    #     dataset=val_data,
    #     batch_size=args.batch_size,
    #     num_workers=num_workers
    # )
    
    pretrained_model = load_checkpoint(args.checkpoint_path, device=args.device)

    all_params = [p[0] for p in pretrained_model.named_parameters()]
    all_params_count = [p[1].numel() for p in pretrained_model.named_parameters()]
    all_params_count_with_grad = [p[1].numel() for p in pretrained_model.named_parameters() if p[1].requires_grad]

    # Parse arguments
    pl.seed_everything(args.seed)
    subspace_method = 'AS'
    result_dir = os.path.join(args.save_dir,subspace_method)
    os.makedirs(result_dir, exist_ok = True)
    
    subunit = [p[0] for p in pretrained_model.named_parameters() if p[1].requires_grad]
    # subunit = [p for p in all_params if p.split('.')[1] in component_to_param_map[subunit_name]]
    # # subunit = ["jtnn_vae.A_assm.weight"]
    
    Np = sum([p[1].numel() for p in pretrained_model.named_parameters() if p[0] in subunit])

    print(f'Total number of parameters: {sum(all_params_count)}')
    print(f'Total number of parameters requiring grad: {Np}')
    
    
    
    model = load_checkpoint(args.checkpoint_path, device=args.device)
    

    lr_init = 1e-1
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init)
    
    
    print(len(train_dataloader))
    sig_nn = 1e-1 #np.round(1/np.sqrt(0.09),3)
    Nsamples = 100
    max_rank = cli_args.AS_dim
    assert max_rank <= Nsamples, f"number of AS dimension needs to be lower than {Nsamples}, number of gradient samples"
    grads = np.zeros((Nsamples, Np))
    
    # if hparams.gpu:
    #     model = model.cuda()
        
    # for p in model.named_parameters():
    #     if p[0] not in subunit:
    #         p[1].requires_grad_(False)
    
    for idx, input_batches in enumerate(train_dataloader):
        if subspace_method in ['AS']:
            init_weights(model, pretrained_model, sig_nn,  subunit)
        else:
            init_weights(model, pretrained_model, sig_nn,  subunit, Np)
        # print(input_batches)
        
        # for p in model.named_parameters():
        #     if p[0] not in subunit:
        #         p[1].requires_grad_(False)
        # print(input_batches.targets())
        one_gradient_sample(model,input_batches,get_loss_func(args), optimizer,args)
        # # torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        
        grad_sample = torch.cat([p[1].grad.flatten() for p in model.named_parameters() if p[0] in subunit]).cpu().detach().numpy()

        grads[idx,:] = grad_sample
        
        if idx == Nsamples-1:
            break
    
    grads /= np.sqrt(Nsamples)    
    # utils.update_hparams(hparams, model)
    print(grads.shape)
    
    print(f'gradient range: [{np.min(grads)},{np.max(grads)}]')
    print(f'percentage of bounded gradient: {round(100*np.sum(~np.isnan(grads))/(Nsamples*Np),3)}')
    
    _, Sigma, Vt = randomized_svd(grads, n_components=max_rank, n_iter=5, random_state=1)
    
    print(f'sigma range: [{np.min(Sigma)},{np.max(Sigma)}]')
    
    #if hparams.benchmark:
    #    np.savez_compressed(os.path.join(result_dir,f'singular_values_{subunit_name}_{hparams.seed}.npz'), S=Sigma, V = Vt, mean = subspace_data['mean'])
    #else:
    np.savez_compressed(os.path.join(result_dir,f'singular_values_{args.seed}_AS_dim_{max_rank}.npz'), S=Sigma, V = Vt)
    print(Sigma)
    
    
