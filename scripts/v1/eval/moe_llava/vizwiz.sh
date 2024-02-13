#!/bin/bash

CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
deepspeed moellava/eval/model_vqa_loader.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL}/vizwiz/test \
    --answers-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 scripts/convert_vizwiz_for_submission.py \
    --annotation-file ${EVAL}/vizwiz/llava_test.jsonl \
    --result-file ${EVAL}/vizwiz/answers/${CKPT_NAME}.jsonl \
    --result-upload-file ${EVAL}/vizwiz/answers_upload/${CKPT_NAME}.json