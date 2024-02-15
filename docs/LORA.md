## Training for LoRA tuning models
Coming soon...

## Evaluation for LoRA tuning models

You can evaluate the model directly after LoRA tuning as [EVAL.md](../docs/EVAL.md).

Or you can evaluate it after merging weights as follows.

### Optional

You can use `script/merge_moe_lora_weights.py` to merge the LoRA weights.

```Shell
deepspeed --include localhost:0 script/merge_lora_weights.py \
	--model-path checkpoints/moellava-phi-moe-lora \
	--save-model-path checkpoints/moellava-phi-moe-merge
```

> [!Warning]
> 🚨 Please do not have `lora` in `--save-model-path` and `lora` should in `--model-path`.


Then evaluate `checkpoints/llavaphi-moe-merge` as [EVAL.md](../docs/EVAL.md)
