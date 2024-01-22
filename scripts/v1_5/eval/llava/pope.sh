#!/bin/bash

CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m moellava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --image-folder ${EVAL}/pope/val2014 \
    --answers-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 moellava/eval/eval_pope.py \
    --annotation-dir ${EVAL}/pope/coco \
    --question-file ${EVAL}/pope/llava_pope_test.jsonl \
    --result-file ${EVAL}/pope/answers/${CKPT_NAME}.jsonl