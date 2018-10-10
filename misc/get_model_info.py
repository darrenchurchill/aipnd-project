#! /usr/bin/env python3
"""Load a checkpoint and print a summary about it."""

import argparse

from context import Classifier


def main():
    parser = argparse.ArgumentParser(
        description=('Load a model and print a summary about it'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('checkpoint', type=str,
                        help='path to the checkpoint to load')
    args = parser.parse_args()

    classifier = Classifier(checkpoint=args.checkpoint)

    for name, value in classifier.summary_info:
        print(f'{name:19}: {value}')


if __name__ == '__main__':
    main()
