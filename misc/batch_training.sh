#! /bin/bash
# Script to perform batch training of different model architectures using
# train.py. Doesn't save checkpoint files to conserve disk space.

USAGE="\
usage: batch_training.sh [-h] [--no_active_session] [--no_gpu]
                         [--save_dir_root SAVE_DIR_ROOT] [data_directory]"

HELP="\
Train each type of model architecture defined in classifier.py and
output training, validation, and testing data to a set of files.

Model hyperparameters can be specified in a file named
hyperparameters.yml located in the SAVE_DIR_ROOT, if used, or the
current directory if not used. The format of the file should follow the
example below. If no file is given, the default values will be used and
written to the hyperparameters.yml file as documentation.

# hyperparameters.yml
learn_rate: 0.001
epochs: 6
hidden_units:
  - 4096
  - 1000

positional arguments:
  data_directory        path to the directory containing the training,
                        validation and testing sets. (default: flowers)

optional arguments:
  -h, --help            show this help message and exit
  --no_active_session   don't keep session alive (if on a local machine)
  --no_gpu              don't use the gpu to train the network
  --save_dir_root SAVE_DIR_ROOT
                        the directory root to save output files into"

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

source misc/bash-yaml/yaml.sh

# Default values
LEARN_RATE=0.001
EPOCHS=6
HIDDEN_UNITS=(4096 1000)
DATA_DIR='flowers' # if none given

# initial arg values
data_dir=''
no_active=''
gpu='--gpu'
save_root='.'

# Parse arguments. Example from:
# https://medium.com/@Drew_Stokes/bash-argument-parsing-54f3b81a6a8f
while (( "$#" )); do
  case "$1" in
    -h|--help)
      print_help $0
      exit 1
      ;;
    --no_active_session)
      no_active='--no_active_session'
      shift
      ;;
    --no_gpu)
      gpu=''
      shift
      ;;
    --save_dir_root)
      save_root=$(echo $2 | cut -f1 -d'/')
      shift 2
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      print_usage $0
      exit 1
      ;;
    *) # preserve positional arguments
      if [ -z "$data_dir" ]  # if data_dir is empty
      then
         data_dir=$(echo $1 | cut -f1 -d'/')
      else
        echo "Error: Too many positional args: $@" >&2
        print_usage $0
        exit 1
      fi
      shift
      ;;
  esac
done

if [ -z "$data_dir" ]
then
  data_dir="$DATA_DIR"
fi

#echo "Data dir: $data_dir"
#echo "no active: $no_active"
#echo "gpu: $gpu"
#echo "save root: $save_root"
#echo
#exit 0

architectures=$(python3 misc/get_model_names.py)

hyperparams="${save_root}/hyperparameters.yml"
if [ -f "$hyperparams" ]
then
  echo "Loading hyperparameters from $hyperparams"
  create_variables "$hyperparams"
  if [ -z "$learn_rate" ]
  then
    echo "No learn_rate set in $hyperparams"
    exit 1
  elif [ -z "$epochs" ]
  then
    echo "No epochs set in $hyperparams"
    exit 1
  elif [ -z "$hidden_units" ]
  then
    echo "No hidden_units set in $hyperparams"
    exit 1
  fi
else
  learn_rate="$LEARN_RATE"
  epochs="$EPOCHS"
  hidden_units="${HIDDEN_UNITS[*]}"

  echo "Writing default hyperparameters to $hyperparams"
  echo "# Example hyperparameters file for batch training" > "$hyperparams"
  echo "learn_rate: $learn_rate" >> "$hyperparams"
  echo "epochs: $epochs" >> "$hyperparams"
  echo "hidden_units:" >> "$hyperparams"
  for value in ${hidden_units[@]}; do
    echo "  - $value" >> "$hyperparams"
  done
fi

#echo "learn_rate: $learn_rate"
#echo "epochs: $epochs"
#echo "hidden_units: ${hidden_units[*]}"
#exit 0

#architectures='alexnet'  # for testing just one model

for arch in $architectures
do
  OUT_FILE="${save_root}/${arch}.txt"
  echo "Training $arch and writing output to $OUT_FILE"
  python3 ./train.py $data_dir $gpu $no_active --test_model \
    --no_save_checkpoint \
    --arch $arch \
    --learning_rate $learn_rate \
    --epochs $epochs \
    --hidden_units $hidden_units \
    --save_dir $save_root \
    --write_log_file
done
