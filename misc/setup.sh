#! /bin/bash
# Download and unzip flowers directory containing training, validation and
# testing datasets.
# If using a Mac, install wget with homebrew (brew install wget) before using
# this script.

directory="flowers"

if [ ! -d "$directory" ]
then
  echo >&2 "Run this script from the root directory of this repo."
  exit -1
fi

flowers_data="flower_data.tar.gz"
wget "https://s3.amazonaws.com/content.udacity-data.com/nd089/$flowers_data"
tar -C "$directory" -xvf "$flowers_data"
rm "$flowers_data"
