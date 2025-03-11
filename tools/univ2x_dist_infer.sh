#!/usr/bin/env bash
# -------------------------- inference.sh --------------------------
T=$(date +%m%d%H%M)

CFG=$1
CKPT=$2
GPUS=$3
OUT_FILE=${4:-"results.pkl"} 

GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
MASTER_PORT=${MASTER_PORT:-28596}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
LOG_DIR=${WORK_DIR}logs/inference/

if [ ! -d ${LOG_DIR} ]; then
    mkdir -p ${LOG_DIR}
fi

echo "Running inference with:"
echo "Config: ${CFG}"
echo "Checkpoint: ${CKPT}"
echo "Output: ${WORK_DIR}${OUT_FILE}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/inference.py \
    $CFG \
    $CKPT \
    --out ${WORK_DIR}${OUT_FILE} \
    --launcher pytorch \
    2>&1 | tee ${LOG_DIR}inference_$T.log

echo "Inference results saved to: ${WORK_DIR}${OUT_FILE}"
