#! /usr/bin/env python3
"""Train a new network on a dataset and save the model as a checkpoint. Prints
out training loss, validation loss, and validation accuracy while the network
trains.

Basic Usage:
python train.py data_directory

Options:
- Set the directory to save checkpoints
  python train.py data_directory --save_dir save_directory
- Choose the model architecture
  python train.py data_directory --arch 'vgg13'
- Set hyperparameters for the model and for training
  python train.py data_directory --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use the GPU for training
  python train.py data_directory --gpu
"""

import argparse
import os
import sys

import torch

from classifier import Classifier
from model_trainer import ModelTrainer
from workspace_utils import active_session


def main():
    # Instantiating with formatter_class argument will make default values print
    # in the help message.
    parser = argparse.ArgumentParser(
        description=('Train a new network on a dataset and save the model as ' +
                     'a checkpoint'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_directory', type=str,
                        help=('path to the directory containing the ' +
                              'training, validation and testing sets.'))
    parser.add_argument('--save_dir', type=str, default='.',
                        help='set the directory to save checkpoints in')
    parser.add_argument('--checkpoint', type=str,
                        help='load a checkpoint to continue training')
    parser.add_argument('--arch', type=str.lower, default='alexnet',
                        choices=[k.lower()
                                 for k in Classifier.IMAGENET_MODELS.keys()],
                        help='choose the model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate to use while training')
    parser.add_argument('--hidden_units', type=int, nargs='+',
                        default=[4096, 1000],
                        help="sizes of the classifier's hidden layers")
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to go through during training')
    parser.add_argument('--no_validate', action='store_true',
                        help=("don't validate using validation set during " +
                              "training"), )
    parser.add_argument('--test_model', action='store_true',
                        help=('use test dataset to test model accuracy after ' +
                              'training'))
    parser.add_argument('--gpu', action='store_true',
                        help=('use the gpu to train the network if one is ' +
                              'available'))
    parser.add_argument('--no_active_session', action='store_true',
                        help="don't keep session alive (if on a local machine)")
    parser.add_argument('--no_save_checkpoint', action='store_true',
                        help=("don't save a checkpoint after training to " +
                              "save disk space"))
    args = parser.parse_args()
    # print(args)

    data_dir = args.data_directory.rstrip('/')
    try:
        num_categories = len([
            d for d in os.listdir(data_dir + '/test') if d.isnumeric()
        ])
    except FileNotFoundError:
        print(f'ERROR: {data_dir} not found.')
        sys.exit(-1)
    except NotADirectoryError:
        print(f'ERROR: {data_dir} is not a directory. ',
              'Saving checkpoint to current dir instead.',
              file=sys.stderr)
    # print(num_categories)

    if args.gpu:
        device = 'cuda'
        if not torch.cuda.is_available():
            print('ERROR: cuda is not available on this machine.',
                  'Use cpu for training instead.',
                  file=sys.stderr)
            sys.exit(-1)
    else:
        device = 'cpu'

    if args.checkpoint:
        trainer = ModelTrainer(
            data_dir,
            classifier=Classifier(checkpoint=args.checkpoint)
        )
    else:
        trainer = ModelTrainer(
            data_dir,
            model_architecture=args.arch,
            output_size=num_categories,
            hidden_layers=args.hidden_units,
            learn_rate=args.learning_rate
        )

    if args.no_active_session:
        trainer.train_classifier(validate=not args.no_validate,
                                 num_epochs=args.epochs,
                                 device=device,
                                 output_file=sys.stdout)
    else:
        with active_session():  # Keep session active on the remote workspace
            trainer.train_classifier(validate=not args.no_validate,
                                     num_epochs=args.epochs,
                                     device=device,
                                     output_file=sys.stdout)

    if not args.no_save_checkpoint:
        save_dir = args.save_dir.rstrip('/')
        try:
            os.listdir(save_dir)
        except FileNotFoundError:
            os.mkdir(save_dir)
        except NotADirectoryError:
            print(f'WARNING: {save_dir} is not a directory. ',
                  'Saving checkpoint to current dir instead.',
                  file=sys.stderr)
            save_dir = '.'

        trainer.classifier.save_checkpoint(save_dir + '/checkpoint.pth')

    if args.test_model:
        if args.no_active_session:
            accuracy = trainer.test_accuracy(device=device,
                                             output_file=sys.stdout)
        else:
            with active_session():
                accuracy = trainer.test_accuracy(device=device,
                                                 output_file=sys.stdout)

        print(f'Test Accuracy: {accuracy*100:.4f}%')


if __name__ == '__main__':
    main()
