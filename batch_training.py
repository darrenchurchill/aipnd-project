#! /usr/bin/env python3
"""Script to perform batch training of different model architectures using
train.py. Doesn't save checkpoint files to conserve disk space.
"""

import argparse
import subprocess
import sys
import yaml

from classifier import Classifier


def load_hyperparameters(file_path):
    """Load a yaml file and return a dictionary.

    Returns dictionary with default hyperparameters if no file is found. Exits
    program if an exception is raised while loading the file or if an item is
    missing from the hyperparameters config file.

    | default_hyperparameters = {
    |     'learn_rate': 0.001,
    |     'epochs': 10,
    |     'hidden_units': [4096, 1000]
    | }

    Args:
        file_path(str): the path to the file to load.

    Returns:
        (dict): the dictionary created after loading the yaml file, or None
            if no file exists.
    """
    default_hyperparameters = {
        'learn_rate': 0.001,
        'epochs': 10,
        'hidden_units': [4096, 1000]
    }

    try:
        with open(file_path, 'r') as f:
            params = yaml.load(f)
    except FileNotFoundError:
        return default_hyperparameters
    except Exception as e:
        print('Exception while loading YAML file:', e, file=sys.stderr)
        sys.exit(-1)

    for key in default_hyperparameters.keys():
        if key not in params:
            print('ERROR: Invalid hyperparameters file. ' +
                  f'No {key} set in {file_path}.',
                  file=sys.stderr)
            sys.exit(-1)

    return params


def main():
    # Instantiating with formatter_class argument will make default values print
    # in the help message.
    parser = argparse.ArgumentParser(
        description=('Train a new network on a dataset and save the model as ' +
                     'a checkpoint'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_directory', type=str, nargs='?',
                        default='flowers',
                        help=('path to the directory containing the training,' +
                              ' validation and testing sets.'))
    parser.add_argument('--no_active_session', action='store_true',
                        help="don't keep session alive (if on a local machine)")
    parser.add_argument('--no_gpu', action='store_true',
                        help=("don't use the gpu to train the network " +
                              "(use the cpu)"))
    parser.add_argument('--save_dir_root', type=str, default='.',
                        help='the directory root to save output files into.')
    args = parser.parse_args()

    architectures = [k.lower() for k in Classifier.IMAGENET_MODELS.keys()]

    hyperparams = load_hyperparameters(
        f'{args.save_dir_root}/hyperparameters.yml'
    )

    learning_rate = hyperparams['learn_rate']
    epochs = hyperparams['epochs']
    hidden_units = [str(unit) for unit in hyperparams['hidden_units']]

    no_act_sess = ['--no_active_session'] if args.no_active_session else []
    gpu = ['--gpu'] if not args.no_gpu else []

    for arch in architectures:
        try:
            print(f'Training {arch} and writing output to ' +
                  f'{args.save_dir_root}/{arch}.txt')
            prog = [f'./train.py',
                    args.data_directory,
                    *no_act_sess,
                    *gpu,
                    '--arch', f'{arch}',
                    '--learning_rate', f'{learning_rate}',
                    '--epochs', f'{epochs}',
                    '--hidden_units', *hidden_units,
                    '--test_model',
                    '--save_dir', args.save_dir_root,
                    '--no_save_checkpoint',
                    '--write_log_file']
            subprocess.run(prog)
        except KeyboardInterrupt:
            continue


if __name__ == '__main__':
    main()
