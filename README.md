# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In
this project, students first develop code for an image classifier built with
PyTorch, then convert it into a command line application.

## Setup
Install requirements into a new conda environment.

`conda env create -f enviroment.yml [-n aipnd]`

Run setup script to download and unzip flowers datasets into flowers directory.

`./misc/setup.sh`

__Note:__ `setup.sh` uses `wget` to download the compressed archive. If on a mac
install `wget` first:

`brew install wget`

## Usage
`train.py`

Train a new image classifier or continue training a classifier using a
previously saved checkpoint.

`predict.py`

Load a saved model checkpoint and predict an image's top K most probable classes.

### Batch training
`misc/batch_training.sh` and `misc/batch_training.py`

Train each of the model architectures using the hyperparameters loaded from a
yaml file. Training output is written to stdout and saved to a logfile for each
architecture. Example yaml file and logs are in `batch_example`

Both scripts take the same options. I just wanted to see which I liked writing
more.

## Cleanup
To save disk space, you can run the cleanup script to remove the three image
datasets from `flowers/`

`./misc/cleanup.sh`
