#! /bin/bash
# Compress files to turn in into an archive

USAGE="\
usage: make_turnin_archive.sh [-h] [-a]"

HELP="\
Compress files to turn in into an archive. By default only the files I need to
turn in will be included.

optional arguments:
  -h, --help            show this help message and exit
  -a, --all             compress everything, for quickly putting stuff onto the
                        remote workspace"

function print_usage() {
  echo "$USAGE"
}

function print_help() {
  print_usage
  echo
  echo "$HELP"
}

if [ ! -d "misc" ]
then
  echo >&2 "Run this script from the root directory of this repo."
  exit -1
fi

other_files=''

# Parse arguments. Example from:
# https://medium.com/@Drew_Stokes/bash-argument-parsing-54f3b81a6a8f
while (( "$#" )); do
  case "$1" in
    -h|--help)
      print_help
      exit 1
      ;;
    -a|--all)
      other_files="$(find misc -depth 1 | grep -v __pycache__)"
      other_files="$other_files README.md LICENSE environment.yml"
      shift
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      print_usage
      exit 1
      ;;
    *) # preserve positional arguments
      echo "Error: no positional arguments taken." >&2
      print_usage
      exit 1
      shift
      ;;
  esac
done

tar cf project.zip *.ipynb *.html assets predict.py train.py cat_to_name.json \
  workspace_utils.py classifier/*.py $other_files
