#!/bin/bash

CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m moellava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/MME/llava_mme.jsonl \
    --image-folder ${EVAL}/MME/MME_Benchmark_release_version \
    --answers-file ${EVAL}/MME/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

cd ${EVAL}/MME

python convert_answer_to_mme.py --experiment $CKPT_NAME

cd eval_tool

python calculation.py --results_dir answers/$CKPT_NAME
