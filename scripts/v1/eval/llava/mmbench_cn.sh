#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"
CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"

python3 -m moellava.eval.model_vqa_mmbench \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mmbench/$SPLIT.tsv \
    --answers-file ${EVAL}/mmbench/answers/$SPLIT/${CKPT_NAME}.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${CONV}

mkdir -p ${EVAL}/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ${EVAL}/mmbench/$SPLIT.tsv \
    --result-dir ${EVAL}/mmbench/answers/$SPLIT \
    --upload-dir ${EVAL}/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT_NAME}
	
