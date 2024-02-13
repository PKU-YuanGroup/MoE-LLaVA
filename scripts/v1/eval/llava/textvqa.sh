#!/bin/bash


CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m moellava.eval.model_vqa_loader \
    --model-path ${CKPT} \
    --question-file ${EVAL}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ${EVAL}/textvqa/train_images \
    --answers-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

python3 -m moellava.eval.eval_textvqa \
    --annotation-file ${EVAL}/textvqa/TextVQA_0.5.1_val.json \
    --result-file ${EVAL}/textvqa/answers/${CKPT_NAME}.jsonl