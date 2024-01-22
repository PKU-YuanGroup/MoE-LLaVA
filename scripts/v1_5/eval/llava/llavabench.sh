#!/bin/bash

CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
python3 -m moellava.eval.model_vqa \
    --model-path ${CKPT} \
    --question-file ${EVAL}/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ${EVAL}/llava-bench-in-the-wild/images \
    --answers-file ${EVAL}/llava-bench-in-the-wild/answers/${CKPT_NAME}.jsonl \
    --temperature 0 \
    --conv-mode ${CONV}

mkdir -p ${EVAL}/llava-bench-in-the-wild/reviews

python3 moellava/eval/eval_gpt_review_bench.py \
    --question ${EVAL}/llava-bench-in-the-wild/questions.jsonl \
    --context ${EVAL}/llava-bench-in-the-wild/context.jsonl \
    --rule moellava/eval/table/rule.json \
    --answer-list ${EVAL}/llava-bench-in-the-wild/answers_gpt4.jsonl \
                  ${EVAL}/llava-bench-in-the-wild/answers/${CKPT_NAME}.jsonl \
    --output ${EVAL}/llava-bench-in-the-wild/reviews/${CKPT_NAME}.jsonl

python3 moellava/eval/summarize_gpt_review.py -f ${EVAL}/llava-bench-in-the-wild/reviews/${CKPT_NAME}.jsonl