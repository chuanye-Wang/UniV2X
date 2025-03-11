#!/usr/bin/env bash

T=$(date +%m%d%H%M)

CFG=$1
RESULT_FILE=$2
EVAL_METRICS=${3:-"bbox"}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
LOG_DIR=${WORK_DIR}logs/evaluation/

if [ ! -d ${LOG_DIR} ]; then
    mkdir -p ${LOG_DIR}
fi

echo "Running evaluation with:"
echo "Config: ${CFG}"
echo "Result file: ${RESULT_FILE}"
echo "Metrics: ${EVAL_METRICS}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/eval.py \
    $CFG \
    --result ${RESULT_FILE} \
    --eval ${EVAL_METRICS} \
    2>&1 | tee ${LOG_DIR}evaluation_$T.log