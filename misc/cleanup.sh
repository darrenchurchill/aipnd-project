#! /bin/bash
# Remove image datasets in this repository to save disk space.

directory="flowers"

if [ ! -d "$directory" ]
then
  echo >&2 "Run this script from the root directory of this repo."
  exit -1
fi

rm -rf "$directory/test" "$directory/train" "$directory/valid"

