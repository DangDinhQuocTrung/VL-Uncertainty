#!/bin/bash
#BSUB -q gpua100
#BSUB -J testing_vlu
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 23:50
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u dida@dtu.dk
#BSUB -N
#BSUB -oo /zhome/05/8/227717/logs/testing_vlu_%J.out
#BSUB -eo /zhome/05/8/227717/logs/testing_vlu_%J.err
set -e

# SEED=$1
# BENCHMARK=$2
# LVLM=$3
# USE_MODEL_MANAGER=$4
# USE_FASTEST=$5
# LLM=$6
# UNCERTAINTY=$7
# INFERENCE_TEMP=$8
# SAMPLING_TEMP=$9
# SAMPLING_TIME=${10}

ROOT_DIR=/work3/dida
USER_DIR=/zhome/05/8/227717

mkdir -p ${ROOT_DIR}/uncertainty/logs

# Load
pwd
nvidia-smi
module load cuda/12.1
module load python3/3.11.9
source /zhome/05/8/227717/venv/medical/bin/activate
# Export
export PYTHONPATH=${PYTHONPATH}:${USER_DIR}/VL-Uncertainty
export HF_HOME=${ROOT_DIR}/cache
export TORCH_HOME=${ROOT_DIR}/cache
cd ${USER_DIR}/VL-Uncertainty

python3 main.py \
    --seed ${SEED} \
    --benchmark ${BENCHMARK} \
    --lvlm ${LVLM} \
    --use_model_manager ${USE_MODEL_MANAGER} \
    --use_fastest ${USE_FASTEST} \
    --llm ${LLM} \
    --uncertainty ${UNCERTAINTY} \
    --inference_temp ${INFERENCE_TEMP} \
    --sampling_temp ${SAMPLING_TEMP} \
    --sampling_time ${SAMPLING_TIME}
