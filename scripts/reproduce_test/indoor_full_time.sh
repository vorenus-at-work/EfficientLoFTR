#!/bin/bash -l
#set -x

SCRIPTPATH=$(dirname $(readlink -f "$0"))
echo "SCRIPTPATH: $SCRIPTPATH"
PROJECT_DIR="${SCRIPTPATH}/../../"

# Activate the conda environment
#CONDA_ENV_NAME="eloftr"
#source ~/anaconda3/bin/conda  # Adjust the path to conda.sh if necessary
#conda activate $CONDA_ENV_NAME

# Check if PYTHONPATH is set, if not, initialize it with the project directory
# if [ -z "$PYTHONPATH" ]; then
#     export PYTHONPATH=$PROJECT_DIR
# else
#     export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
# fi

echo "PYTHONPATH: $PYTHONPATH"  # Print the PYTHONPATH for verification
echo "Using Python interpreter: $(which python)"  # Print the Python interpreter for verification


cd $PROJECT_DIR

main_cfg_path="configs/loftr/eloftr_full.py"

profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

ckpt_path="weights/eloftr_outdoor.ckpt"

dump_dir="dump/eloftr_full_scannet"
data_cfg_path="configs/data/scannet_test_1500.py"

echo "Pausing before running Python script. Press Enter to continue..."
read

python ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark \
    --scannetX '640' \
    --scannetY '480' \
    --rmbd 0 \
    --thr 0.1 \
    --ransac_times 1
