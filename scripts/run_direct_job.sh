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

ROOT_DIR=/work3/dida
USER_DIR=/zhome/05/8/227717

mkdir -p ${ROOT_DIR}/uncertainty/logs

# Load
pwd
module load cuda/12.1
module load python3/3.11.9
source /zhome/05/8/227717/venv/health/bin/activate
# Export
export PYTHONPATH=${PYTHONPATH}:${USER_DIR}/VL-Uncertainty
export HF_HOME=${ROOT_DIR}/cache
export TORCH_HOME=${ROOT_DIR}/cache
export CUDA_VISIBLE_DEVICES=0
cd ${USER_DIR}/VL-Uncertainty

# python3 main.py --uncertainty semantic_entropy_nli --compute_visual_statistics True
python3 main.py --uncertainty euq --compute_visual_statistics True
python3 update_llm_judge.py --dataset ViLP
