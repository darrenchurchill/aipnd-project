#! /usr/bin/env python3
"""ModelTrainer class definition. Given a Classifier to train and a directory of
training, validating and testing datasets, the Trainer trains the Classifier's
model.
"""

import torch.utils.data
from torchvision import datasets, transforms

from classifier import Classifier


class ModelTrainer(object):
    """Convenience class to train a PyTorch model using DataLoaders.

    Attributes:
        classifier (classifier.Classifier): the Classifier object containing the
            model to be trained.
    """
    def __init__(self, dataset_root='flowers', classifier=None,
                 **classifier_kwargs):
        """You can create a ModelTrainer with a Classifier object if you have a
        model checkpoint to load and want to continue training, or you can pass
        a valid set of keyword arguments to create a new Classifier object
        before training.

        Args:
            dataset_root (str): the directory where the train, valid, and test
                datasets are located from.
            classifier (classifier.Classifier): the Classifier containing the
                PyTorch model to train. If no Classifier is given as an
                argument, a new Classifier can be created using the
                classifier_kwargs passed instead.
            **classifier_kwargs: if no classifier is given, these will be used
                to create a new Classifier to train. See Classifier.__init__()
                in classifier.py for valid kwargs.

        Examples:
            trainer = ModelTrainer(
                'flowers',
                classifier=Classifier(checkpoint='checkpoint.pth')
            )

            trainer = ModelTrainer(
                'flowers',
                model_architecture='alexnet',
                output_size=102,
                hidden_layers=[4096, 1000],
                learn_rate=0.005
            )
        """
        self._data_dir = dataset_root
        self._train_dir = self._data_dir + '/train'
        self._valid_dir = self._data_dir + '/valid'
        self._test_dir = self._data_dir + '/test'

        # Define transforms for training, validation and testing sets.
        # validation and test transforms are the same.
        self._data_transforms = {
            'train': transforms.Compose([transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             [0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]
                                         )]),
            'valid': transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             [0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225]
                                         )]),
            'test': transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225]
                                        )])
        }

        # Load the datasets with ImageFolder
        self._image_datasets = {
            'train': datasets.ImageFolder(
                self._train_dir,
                transform=self._data_transforms['train']
            ),
            'valid': datasets.ImageFolder(
                self._valid_dir,
                transform=self._data_transforms['valid']
            ),
            'test': datasets.ImageFolder(
                self._test_dir,
                transform=self._data_transforms['test']
            )
        }

        # Using the image datasets and the transforms, define the dataloaders
        self._dataloaders = {
            'train': torch.utils.data.DataLoader(self._image_datasets['train'],
                                                 batch_size=64, shuffle=True),
            'valid': torch.utils.data.DataLoader(self._image_datasets['valid'],
                                                 batch_size=64),
            'test': torch.utils.data.DataLoader(self._image_datasets['test'],
                                                batch_size=64)
        }

        if classifier:
            self.classifier = classifier
        else:
            self.classifier = Classifier(
                class_to_idx=self._image_datasets['train'].class_to_idx,
                **classifier_kwargs
            )

    def train_classifier(self, validate=True, num_epochs=3, print_every=40,
                         device='cuda', output_file=None):
        """Train the classifier and validate every 'print_every' steps through
        each epoch.

        Args:
            validate (bool): whether to use validation set to validate the model
                while training. The training goes faster if you don't validate.
            num_epochs (int): See Classifier.train_classifier()
            print_every (int): See Classifier.train_classifier()
            device (str): See Classifier.train_classifier()
            output_file (io.TextIOWrapper): See Classifier.train_classifier()

        Returns:
            None
        """
        validloader = self._dataloaders['valid'] if validate else None
        self.classifier.train_classifier(self._dataloaders['train'],
                                         validloader,
                                         num_epochs,
                                         print_every,
                                         device,
                                         output_file)

    def test_accuracy(self, device='cuda', output_file=None):
        """Test and calculate the classifier's accuracy using the testing set.

        Args:
            device (str): see Classifier.validate()
            output_file (io.TextIOWrapper): see Classifier.validate()

        Returns:
            (float): the model's accuracy in classifying images in the testing
                set.
        """
        if output_file is not None:
            print('Testing model accuracy: ', end='',
                  file=output_file, flush=True)

        accuracy = self.classifier.validate(self._dataloaders['test'],
                                            device,
                                            output_file)[1]

        # Clear the final 'Processing batch' message from screen
        if output_file is not None:
            print('\r                                                       \r',
                  end='', file=output_file, flush=True)

        return accuracy
