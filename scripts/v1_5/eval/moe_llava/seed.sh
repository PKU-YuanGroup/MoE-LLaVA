#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CONV="conv_template"
CKPT_NAME="your_ckpt_name"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"

for IDX in $(seq 0 $((CHUNKS-1))); do
    deepspeed --include localhost:${GPULIST[$IDX]} --master_port $((${GPULIST[$IDX]} + 29501)) moellava/eval/model_vqa_loader.py \
        --model-path ${CKPT} \
        --question-file ${EVAL}/seed_bench/llava-seed-bench.jsonl \
        --image-folder ${EVAL}/seed_bench \
        --answers-file ${EVAL}/seed_bench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode ${CONV} &
done

wait

output_file=${EVAL}/seed_bench/answers/${CKPT_NAME}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${EVAL}/seed_bench/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ${EVAL}/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file ${EVAL}/seed_bench/answers_upload/${CKPT_NAME}.jsonl						 
					
