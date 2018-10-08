#! /usr/bin/env python3
"""Predict a flower name along with the probability of that name.
Basic Usage: python predict.py /path/to/image checkpoint
Options:
- Return top K most likely classes:
  python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names:
  python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference:
  python predict.py input checkpoint --gpu
"""

import argparse
import json
import os
import random
import sys

import torch

from classifier import Classifier


def get_random_image_from_dir(directory):
    """Get a random image path from a directory.

    | dir is expected to be structured like:
    |
    | dir
    | ├── 1
    | │   ├── image_06743.jpg
    | │   ├── image_06752.jpg
    | ...
    | └── 99
    |     ├── image_07833.jpg
    |     ├── image_07838.jpg
    |     └── image_07840.jpg

    Args:
        directory (str): the directory containing images

    Returns:
        (str): path of the randomly chosen image in dir/*/*
    """
    image_dirs = [os.path.join(directory, file)
                  for file in os.listdir(directory)
                  if file.isnumeric()]

    image_dir = random.choice(image_dirs)

    images = [os.path.join(image_dir, file)
              for file in os.listdir(image_dir)]

    return random.choice(images)


def load_json(file_path):
    """Load a json file and return a dictionary.
    Exits program if an exception is raised while loading the file.

    Args:
        file_path (str): the path to the file to load.

    Returns:
        (dict): the dictionary created after loading the json file.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'ERROR: file {file_path} not found.',
              file=sys.stderr)
        sys.exit(-1)
    except Exception as e:
        print('Exception while loading JSON file:', e, file=sys.stderr)
        sys.exit(-1)


def main():
    # Instantiating with formatter_class argument will make default values print
    # in the help message.
    parser = argparse.ArgumentParser(
        description='Process an image & report results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('image_path', type=str,
                        help=('path to the image to process or to a dataset ' +
                              'directory with images to choose randomly from ' +
                              'Ex: flowers/test/1/image_06743.jpg or ' +
                              'flowers/test'))
    parser.add_argument('checkpoint', type=str,
                        help='path to the model checkpoint to load')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str,
                        help='use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help=('if available, use gpu to process the image ' +
                              'instead of the cpu'))
    args = parser.parse_args()
    # print(args)

    if os.path.isdir(args.image_path):
        print(f'{args.image_path} is a directory.',
              'Choosing a random image to process.')
        image_path = get_random_image_from_dir(args.image_path)
        print(f'Using image: {image_path}')
    else:
        image_path = args.image_path

    if not os.path.isfile(args.checkpoint):
        print(f'ERROR: {args.checkpoint} is not a file.', file=sys.stderr)
        sys.exit(-1)

    if args.category_names:
        cat_to_name = load_json(args.category_names)
    else:
        cat_to_name = None

    if args.gpu:
        device = 'cuda'
        if not torch.cuda.is_available():
            print('ERROR: cuda is not available on this machine.',
                  'Use cpu for prediction instead.',
                  file=sys.stderr)
            sys.exit(-1)
    else:
        device = 'cpu'

    classifier = Classifier(checkpoint=args.checkpoint)
    probs, classes = classifier.predict(image_path,
                                        topk=args.top_k,
                                        device=device)

    if cat_to_name:
        classes = [cat_to_name[c] for c in classes]
        class_len = len(max(cat_to_name.values(), key=len))
    else:
        class_len = 10  # padding needed to space column 1 title 'Class' below

    # print(probs)
    # print(classes)

    print(f'{"Class":{class_len}}{"Probability"}')
    for prob, class_ in zip(probs, classes):
        print(f'{class_:{class_len}}{prob:4.2f}')


if __name__ == '__main__':
    main()
