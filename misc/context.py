"""Explicitly modify sys.path to resolve Classifier imports for python files
located in this directory.
"""

import os
import sys

# print('sys.path:', sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print('sys.path:', sys.path)

from classifier.classifier import Classifier
import train
