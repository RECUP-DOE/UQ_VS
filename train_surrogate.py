import numpy as np
import chemprop
from chemprop.data import get_data, get_task_names
from chemprop.utils import create_logger
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prop_name', default = 'DRD2',type=str ,help='property name')
    parser.add_argument('--epochs', default = 20,type=int ,help='training epochs')
    parser.add_argument('--batch_size', default = 64,type=int ,help='batch size')
    parser.add_argument('--num_workers', default = 2,type=int ,help='number of workers')
    cli_args = parser.parse_args()


    arguments = [
        '--data_path', f'data/{cli_args.prop_name.lower()}/train_split.csv',
    #    '--separate_val_path', 'data/jnk3/validation_split.csv',
        '--separate_test_path', f'data/{cli_args.prop_name.lower()}/test_split.csv',
        '--dataset_type', 'classification',
        '--save_dir', f'{cli_args.prop_name.lower()}_checkpoints/basic',
        '--metric', 'precision',
        #'--extra_metrics', 'recall,precision,accuracy',
        '--quiet',
        '--epochs', f'{cli_args.epochs}',
        '--batch_size', f'{cli_args.batch_size}',
        '--num_workers', f'{cli_args.num_workers}',
    #    '--features_generator', 'rdkit_2d_normalized', '--no_features_scaling',
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

    data = get_data(
            path=args.data_path,
            args=args,
            logger=logger,
            skip_none_targets=True,
            data_weights_path=args.data_weights_path
        )
    args.features_size = data.features_size()

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

    test_score = chemprop.train.run_training(args=args, data=data)

    print(test_score)