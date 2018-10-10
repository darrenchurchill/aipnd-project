#! /usr/bin/env python3
"""Print the available model architectures from classifier.py
Used to get valid architecture names to do batch training from shell script.
"""

from context import Classifier


def main():
    for arch in Classifier.IMAGENET_MODELS.keys():
        print(arch.lower())


if __name__ == '__main__':
    main()
