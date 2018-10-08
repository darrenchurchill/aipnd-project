#! /bin/bash
# Download and unzip flowers directory containing training, validation and
# testing datasets.
# If using a Mac, install wget with homebrew (brew install wget) before using
# this script.


flowers_data="flower_data.tar.gz"
wget "https://s3.amazonaws.com/content.udacity-data.com/nd089/$flowers_data"
tar -C "flowers" -xvf "$flowers_data"
rm "$flowers_data"
