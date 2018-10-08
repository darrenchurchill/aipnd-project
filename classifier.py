#! /usr/bin/env python3
"""Classifier class definition """

import sys
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import IO

import numpy as np

from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models


class Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        """Build a feedforward network with an arbitrary number of hidden layers.

        Args:
            input_size (int): the size of the input layer.
            output_size (int): the size of the output layer.
            hidden_layers (list[int]): the sizes of the hidden layers.
            drop_p (float): the node dropout frequency to use during training
        """
        super().__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers[0])]
        )

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(
            [nn.Linear(h1, h2) for h1, h2 in layer_sizes]
        )

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """Forward pass through the network, returns the output logits."""
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


class Classifier(object):
    """Classifier containing a pretrained PyTorch model used for transfer
    learning.

    The Classifier's pretrained model is one in Classifier.IMAGENET_MODELS whose
    classifier will be replaced with a new classifier. The model's pretrained
    network serves as a feature detector providing inputs to the new classifer
    network.

    Attributes:
        input_size (int): the number of inputs being given to the model's
            classifier.
        output_size (int): the number of categories this Classifier can
            classify.
        hidden_layers (list[int]): the size of each hidden layer in the model's
            classifier network.
        learn_rate (float): the learning rate to use while training.
        drop_p (float): the node dropout frequency to use when training the
            classifier network.
        model: the PyTorch model being used.
        model.current_epoch (int): the current training epoch.
        model_architecture (str): the pretrained model's architecture.
    """
    # Pretrained models:
    # Only Alexnet, VGG, and DenseNet architectures seem compatible with the
    # transfer learning method involving replacing the final classifier.
    IMAGENET_MODELS = {
        'ALEXNET': models.alexnet,
        'VGG11': models.vgg11,
        'VGG11_BN': models.vgg11_bn,
        'VGG13': models.vgg13,
        'VGG13_BN': models.vgg13_bn,
        'VGG16': models.vgg16,
        'VGG16_BN': models.vgg16_bn,
        'VGG19': models.vgg19,
        'VGG19_BN': models.vgg19_bn,
        'DENSENET121': models.densenet121,
        'DENSENET161': models.densenet161,
        'DENSENET169': models.densenet169,
        'DENSENET201': models.densenet201,
    }
    # not_working {
    #     'RESNET18': models.resnet18,
    #     'RESNET34': models.resnet34,
    #     'RESNET50': models.resnet50,
    #     'RESNET101': models.resnet101,
    #     'RESNET152': models.resnet152,
    #     'SQUEEZENET1_0': models.squeezenet1_0,
    #     'SQUEEZENET1_1': models.squeezenet1_1,
    #     'INCEPTION_V3': models.inception_v3,
    # }

    def __init__(self, output_size=None, hidden_layers=None, learn_rate=0.001,
                 drop_p=0.5, checkpoint=None, model_architecture=None,
                 class_to_idx=None):
        """Set up a transfer learning model using a pretrained CNN model as a
        feature detector with a classifier added to be trained on a new dataset.

        A Classifier can be created by either:
        1.  Specifying a saved PyTorch checkpoint file to load a previously
            trained transfer learning model.
        2.  Specifying a pretrained model architecture to use as a feature
            detector, along with the classifier's output_size, hidden_layers
            sizes, learn_rate, node drop_p, and class_to_idx mapping.

        Args:
            output_size (int): the number of categories in the dataset
            hidden_layers (list[int]): the sizes of each hidden layer in the
                classifier that's going to be trained on the dataset
            learn_rate (float): the learn rate to use while training the
                classifier.
            drop_p (float): the node dropout frequency to use when training the
                classifier.
            checkpoint (str): the path to the PyTorch checkpoint file to load.
                All other arguments will be ignored if this argument is given.
            model_architecture (str): the PyTorch model to use as the feature
                detector. Must match a key in Classifier.IMAGENET_MODELS.
            class_to_idx (dict): mapping of integer classes to PyTorch Tensor
                indices. This mapping can be found as a PyTorch ImageFolder
                attribute of the same name for either the training, validating,
                or testing datasets.

        Examples:
            c = Classifier(checkpoint='checkpoint.pth')

            OR

            from torchvision import datasets, transforms

            training_transforms = transforms.Compose(
                [transforms.RandomRotation(30),
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])]
            )

            training_folder = datasets.ImageFolder(
                self.train_dir,
                transform=training_transforms
            )

            c = Classifier(model_architecture='vgg16',
                           output_size=102,
                           hidden_layers=[4096, 1000],
                           learn_rate=0.01,
                           drop_p=0.4,
                           class_to_idx=training_folder.class_to_idx)
        """
        if checkpoint is not None:
            # The map_location argument allows object deserialization on a
            # CPU-only machine even when the checkpoint was saved on a CUDA
            # enabled machine without first moving the model back to the CPU.
            keyword = {}
            if not torch.cuda.is_available():
                keyword['map_location'] = 'cpu'
            checkpoint = torch.load(checkpoint, **keyword)
            self.__setup_model(**checkpoint)

        else:
            if output_size is None:
                print('ERROR: output_size cannot be None when building a',
                      'new model.',
                      file=sys.stderr)
                sys.exit(-1)
            if hidden_layers is None:
                print('ERROR: hidden_layers cannot be None when building a',
                      'new model.',
                      file=sys.stderr)
                sys.exit(-1)

            self.__setup_model(output_size=output_size,
                               hidden_layers=hidden_layers,
                               learn_rate=learn_rate,
                               drop_p=drop_p,
                               model_architecture=model_architecture,
                               class_to_idx=class_to_idx)

    def __setup_model(self, **kwargs):
        """Helper to Classifier.__init__()
        Setup the Classifier's model using checkpoint information or the
        information to load a new model and classifier for training.

        Keyword Args:
        Will always be called with the following, which is enough information
        to build load a new model and add a classifier to be trained:
        - model_architecture
        - output_size
        - hidden_layers
        - learn_rate
        - drop_p
        - class_to_idx
        If the following are passed to this function, the checkpoint state will
        be loaded so the model can be used to classify images or so training
        can continue.
        - input_size
        - current_epoch
        - model_state_dict
        - optimizer_state_dict
        """
        self.model_architecture = kwargs['model_architecture'].upper()
        self.model = Classifier.IMAGENET_MODELS[self.model_architecture](
            pretrained=True
        )

        if 'input_size' in kwargs:  # Loading from a checkpoint
            self.input_size = kwargs['input_size']
            self.model.current_epoch = kwargs['current_epoch']

        else:  # No checkpoint, will be creating a new classifier for the model
            # The number of features coming from the feature detector CNN
            # print('Classifier.model.classifier:')
            if 'ALEXNET' in self.model_architecture:
                # print(self.model.classifier)
                self.input_size = self.model.classifier[1].in_features
            elif 'VGG' in self.model_architecture:
                # print(self.model.classifier)
                self.input_size = self.model.classifier[0].in_features
            elif 'DENSENET' in self.model_architecture:
                # print(self.model.classifier)
                self.input_size = self.model.classifier.in_features
            # The structure of the 3 model architectures below don't seem
            # compatible with the training code in the rest of this file.
            # Errors show up during the forward or backward pass.
            # elif ('RESNET' in self.model_architecture or
            #       'INCEPTION' in self.model_architecture):
                # print(self.model.fc)
                # self.input_size = self.model.fc.in_features
                # self.model.classifier = self.model.fc
            # elif 'SQUEEZENET' in self.model_architecture:
            #     print(self.model.classifier)
            #     self.input_size = self.model.classifier[1].in_channels

            # print('Input size:', self.input_size)

            # Freeze the feature detector parameters to prevent backpropagating
            # through them.
            for param in self.model.parameters():
                # print(param)
                param.requires_grad = False

            self.model.current_epoch = 1

        self.output_size = kwargs['output_size']
        self.hidden_layers = kwargs['hidden_layers']
        self.learn_rate = kwargs['learn_rate']
        self.drop_p = kwargs['drop_p']

        self.model.class_to_idx = kwargs['class_to_idx']
        self.model.classifier = Network(self.input_size,
                                        self.output_size,
                                        self.hidden_layers,
                                        self.drop_p)

        if 'model_state_dict' in kwargs:  # load the state from checkpoint
            self.model.load_state_dict(kwargs['model_state_dict'])

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(),
                                    lr=self.learn_rate)

        if 'optimizer_state_dict' in kwargs:  # load the state from checkpoint
            self.optimizer.load_state_dict(kwargs['optimizer_state_dict'])

    def save_checkpoint(self, checkpoint_path='checkpoint.pth'):
        """Save the current model state as a PyTorch checkpoint so it can
        be loaded later.

        Args:
            checkpoint_path (str): the location to save the checkpoint. Should
                be a valid relative or absolute path.

        Returns:
            None
        """
        # Move the model back to the cpu so it can be loaded onto machines
        # without gpu's as well.
        self.model.to('cpu')

        checkpoint = {
            'model_architecture': self.model_architecture,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'hidden_layers': self.hidden_layers,
            'learn_rate': self.learn_rate,
            'drop_p': self.drop_p,
            'class_to_idx': self.model.class_to_idx,
            'current_epoch': self.model.current_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def validate(self, dataloader, device='cuda', print_status=False):
        """Validate the model using the dataloader and calculate the loss and
        accuracy.

        Args:
            dataloader (DataLoader): the PyTorch DataLoader containing the
                images to use for validation.
            device (str): the device to use to perform the PyTorch operations.
                Should be either 'cuda' or 'cpu'.
            print_status (bool): whether to print additional status messages
                to stdout. These messages can help monitor progress in the
                terminal while training.

        Returns:
            (float, float): the validation loss, the validation accuracy.
        """
        test_loss = 0
        accuracy = 0
        num_images = len(dataloader)

        self.model.to(device)

        # Put model in eval mode to turn off node dropouts
        self.model.eval()

        if print_status:
            print('Processing batch: ', end='', flush=True)

        with torch.no_grad():
            num_batches = len(dataloader)
            for step, (images, labels) in enumerate(dataloader, 1):
                if print_status:
                    print(f'{step:3}/{num_batches:3}', end='', flush=True)

                images, labels = images.to(device), labels.to(device)

                output = self.model.forward(images)
                test_loss += self.criterion(output, labels).item()

                ps = torch.exp(output)
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()

                if print_status:
                    # move cursor back 7 spaces for next step msg
                    print('\b' * 7, end='')

        # Put model back into training mode
        self.model.train()

        return test_loss / num_images, accuracy / num_images

    def train_classifier(self, trainloader, validloader=None, num_epochs=3,
                         print_every=40, device='cuda', output_file=None,
                         print_status=False):
        """Train the classifier using the provided dataloaders. Validation set
        will be used every 'print_every' batches during training if one is
        given.

        Args:
            trainloader (DataLoader): the PyTorch DataLoader
                containing the training set.
            validloader (DataLoader): the PyTorch DataLoader
                containing the validation set. Validation will be skipped if
                no validloader is provided.
            num_epochs (int): Number of training epochs to perform.
            print_every (int): How often to print the training status.
            device (str): the device to use to perform the PyTorch operations.
                Should be either 'cuda' or 'cpu'.
            output_file (IO): File-like object to log status info
                to. No info will be written if output_file is none.
            print_status (bool): whether to print additional status messages
                to stdout. These messages can help monitor progress in the
                terminal while training. These status messages contain the same
                Training Loss, Validation Loss and Validation Accuracy
                information as what would be written to the output file, but
                some additional status messages are printed using this option to
                get more frequent updates.

        Returns:
            None
        """
        self.model.to(device)
        start_epoch = self.model.current_epoch
        end_epoch = start_epoch + num_epochs - 1

        for e in range(start_epoch, end_epoch + 1):
            running_loss = 0
            num_batches = len(trainloader)

            for step, (inputs, labels) in enumerate(trainloader, 1):
                # if output_file is not None:
                if print_status:
                    print(f'\rEpoch: {e:{len(str(end_epoch))}}/{end_epoch} |',
                          f'Processing batch: {step:3}/{num_batches:3}',
                          end='', flush=True)
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                # Do the forward and backward passes
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                # loss.requires_grad = True  # needed for certain models???
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if step % print_every == 0:
                    msg = [
                        f'Epoch: {e:{len(str(end_epoch))}}/{end_epoch}   ',
                        f'Training Loss: {running_loss/print_every:.3f}   '
                    ]
                    if validloader is not None:
                        # if output_file is not None:
                        if print_status:
                            print(' | Validating model: ', end='', flush=True)

                        validation_loss, validation_accuracy = \
                            self.validate(validloader, device=device,
                                          print_status=print_status)
                        msg.extend((
                            f'Validation Loss: {validation_loss:.3f}   ',
                            f'Validation Accuracy: {validation_accuracy:.3f}'
                        ))

                    if print_status:
                        print('\r', end='')
                        print(*msg)
                    if output_file is not None:
                        # write to file and flush so it can be seen before
                        # the file is closed. Allows tail -f monitoring
                        print(*msg, file=output_file, flush=True)

                    running_loss = 0

        # Clear the final 'Processing batch' message from screen
        if print_status:
            print('\r                                                         ',
                  flush=True)
        self.model.current_epoch = start_epoch + num_epochs

    def process_image(self, image):
        """Scales, crops, and normalizes a PIL image for a PyTorch model.

        Args:
            image (PIL.Image.Image): the PIL image to process.

        Returns:
            (numpy.ndarray): the processed image as a numpy array.
        """
        min_ = np.min(image.size)
        max_ = np.max(image.size)

        # Find dimension of the longest side if the shortest side is resized to
        # 256 pixels.
        dimension = 256 * max_ / min_

        # Resize to dimension x dimension max resolution and preserve the
        # aspect ratio.
        image.thumbnail((dimension, dimension))

        crop_dim = 224  # crop dimension 224 x 224 pixels
        # Find center crop coordinates
        width, height = image.size
        left = (width - crop_dim) / 2
        top = (height - crop_dim) / 2
        right = (width + crop_dim) / 2
        bottom = (height + crop_dim) / 2

        np_image = np.array(image.crop((left, top, right, bottom))) / 255

        means = np.array([0.485, 0.456, 0.406])
        std_devs = np.array([0.229, 0.224, 0.225])

        np_image = (np_image - means) / std_devs
        np_image = np_image.transpose((2, 0, 1))

        return np_image

    def predict(self, image_path, topk=5, device='cpu'):
        """Predict the class (or classes) of an image using a trained deep
        learning model.

        Args:
            image_path (str): the path to the image to classify.
            topk (int): the number of categories to return predictions for.
            device (str): the device to use when processing the image in the
                model.

        Returns:
            (list[float], list[str]): the probabilities of the top K most likely
                classes, the top K most likely classes.
        """
        self.model.to(device)
        self.model.eval()

        image = Image.open(image_path)
        np_image = self.process_image(image)
        image.close()
        image = np_image

        with torch.no_grad():
            image = torch.from_numpy(image).float()
            image = image.to(device)
            # reshape image to match shapes of images used from dataloaders
            image = image.view(1, *image.shape)
            output = self.model.forward(image)
            # put output back on cpu before moving to numpy
            output = output.cpu()

        values, indices = torch.topk(output.data, topk)
        ps = np.atleast_1d(torch.exp(values).numpy().squeeze()).tolist()

        idx_to_class = {
            value: key for key, value in self.model.class_to_idx.items()
        }
        classes = [idx_to_class[i]
                   for i in np.atleast_1d(indices.numpy().squeeze())]

        return ps, classes
