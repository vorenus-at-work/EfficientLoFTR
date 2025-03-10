#!/bin/bash -l
SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# Check if PYTHONPATH is set, if not, initialize it with the project directory
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=$PROJECT_DIR
else
    export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
fi

echo "PYTHONPATH: $PYTHONPATH"  # Print the PYTHONPATH for verification

cd $PROJECT_DIR

main_cfg_path="configs/loftr/eloftr_optimized.py"

profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

ckpt_path="weights/eloftr_outdoor.ckpt"

dump_dir="dump/eloftr_full_scannet"
data_cfg_path="configs/data/scannet_test_1500.py"
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
    --rmbd 1 \
    --thr 20 \
    --ransac_times 1