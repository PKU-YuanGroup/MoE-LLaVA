#!/bin/bash

CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m moellava.eval.model_vqa \
    --model-path ${CKPT} \
    --question-file ${EVAL}/mm-vet/llava-mm-vet.jsonl \
    --image-folder ${EVAL}/mm-vet/images \
    --answers-file ${EVAL}/mm-vet/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

mkdir -p ${EVAL}/mm-vet/results

python3 scripts/convert_mmvet_for_eval.py \
    --src ${EVAL}/mm-vet/answers/${CKPT_NAME}.jsonl \
    --dst ${EVAL}/mm-vet/results/${CKPT_NAME}.json


python3 moellava/eval/eval_gpt_mmvet.py \
    --mmvet_path ${EVAL}/mm-vet \
    --ckpt_name ${CKPT_NAME} \
    --result_path ${EVAL}/mm-vet/results
