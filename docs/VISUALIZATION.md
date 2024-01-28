## Visualization
Please note that this tutorial is **for MoE models only**.

### Getting expert logits

For visualization, the first step is to get the logits of the experts. GQA and VQAv2 are not currently supported as they generally require multiple GPUs to run. Please change to single GPU if needed.

#scienceqa

In [EVAL.md](https://github.com/PKU-YuanGroup/MoE-LLaVA/blob/main/docs/EVAL.md) we describe how to perform validation. Then, we just need to add `--return_gating_logit "phi_sciqa"` to get the expert logits on ScienceQA benchmark.

```Bash
cd ~/MoE-LLaVA
CKPT_NAME="MoE-LLaVA-Phi2-2.7B-4e"
CKPT="checkpoints/${CKPT_NAME}"
EVAL="eval"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include localhost:0 moellava/eval/model_vqa_science.py \
    --model-path ${CKPT} \
    --question-file ${EVAL}/scienceqa/llava_test_CQM-A.json \
    --image-folder ${EVAL}/scienceqa/images/test \
    --answers-file ${EVAL}/scienceqa/answers/${CKPT_NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi \
    --return_gating_logit "phi_sciqa"  
```

### Distribution of expert loadings


### Distribution of modalities across different experts


### Activated pathways
