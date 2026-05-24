#!/bin/bash
set -e

declare -a seeds=(0)

for seed in ${seeds[@]}; do
    export SEED=${seed}
    export BENCHMARK=ViLP
    export LVLM=llava-1.5-7b-hf
    export USE_MODEL_MANAGER=False
    export USE_FASTEST=False
    export LLM=Qwen2.5-3B-Instruct

    export UNCERTAINTY=vl_uncertainty
    export INFERENCE_TEMP=0.1
    export SAMPLING_TEMP=1.0
    export SAMPLING_TIME=5
    bsub < ./vlu_test_job.sh

    export UNCERTAINTY=vauq
    export INFERENCE_TEMP=0.0
    bsub < ./vlu_test_job.sh

    export UNCERTAINTY=svar
    export USE_MODEL_MANAGER=True
    export INFERENCE_TEMP=0.0
    bsub < ./vlu_test_job.sh
    export USE_MODEL_MANAGER=False

    export BENCHMARK=MisbehaviorBench
    export LVLM=Qwen2.5-VL-7B-Instruct
    export USE_FASTEST=True
    export UNCERTAINTY=euq
    export INFERENCE_TEMP=0.0
    bsub < ./vlu_test_job.sh
    export USE_FASTEST=False
done
