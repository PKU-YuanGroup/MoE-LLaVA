## Visualization

Please note that this tutorial is **for MoE models only**.

### Getting expert logits

For visualization, the first step is to get the logits of the experts. GQA and VQAv2 are not currently supported as they generally require multi-GPUs to run. Please change to single GPU if needed.

In [EVAL.md](https://github.com/PKU-YuanGroup/MoE-LLaVA/blob/main/docs/EVAL.md) we describe how to perform validation. Then, for example, we just need to add `--return_gating_logit "phi_sciqa"` to get the expert logits on ScienceQA benchmark.

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
    --return_gating_logit "phi_sciqa"  # add this command
```

Then, you will get ``phi_sciqa.pt``. Now you can try the other benchmarks through `--return_gating_logit`.

### Distribution of expert loadings

```
python moellava/vis/vis1.py --input phi_sciqa.pt
```

![image](https://github.com/PKU-YuanGroup/MoE-LLaVA/assets/62638829/0a908801-b24a-4e0d-9537-1383c20ea36e)

### Distribution of modalities across different experts

```
python moellava/vis/vis2.py --input phi_sciqa.pt
```

![image](https://github.com/PKU-YuanGroup/MoE-LLaVA/assets/62638829/f1e686ef-ecd5-4b21-a096-fa93c3ef4ae2)

### Activated pathways

```
pip install mplsoccer
python moellava/vis/vis3.py --input phi_sciqa.pt
```

![image](https://github.com/PKU-YuanGroup/MoE-LLaVA/assets/62638829/7f952f7d-2f2d-47d3-80d5-ca733e422aaa)
