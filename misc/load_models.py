#! /usr/bin/env python3
"""Test file to make sure all model architectures load properly and work
with a classifier made of fully connected layers.
"""
import os

from context import Classifier


def main():
    models = Classifier.IMAGENET_MODELS
    dir_root = os.environ['HOME'] + '/.torch/models'

    for arch in models.keys():
        c = Classifier(output_size=102, hidden_layers=[4096, 1000],
                       model_architecture=arch, class_to_idx={})

        # remove downloaded model to conserve disk space
        for f in os.listdir(dir_root):
            os.remove(f'{dir_root}/{f}')


if __name__ == '__main__':
    main()