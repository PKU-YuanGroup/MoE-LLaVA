

- ## **The most important thing**, make sure you understand the behavior of the tokenizer. 
- ## We provide some samples on how different tokenizer behaviors should be changed. 
- ## At the end it describes how to convert LLaVA style models to the MoE architecture.

## Don't have special tokens, but can add special tokens

For those tokenizers that don't have special tokens, but can add special tokens, such as QWenTokenizer or PhiTokenizer. You need to add special tokens.

### QWenTokenizer

Insert the following code after initializing the tokenizer:
```python
tokenizer.add_special_tokens({
    'unk_token': '<|extra_0|>',
    'eos_token': '<|endoftext|>'
})
```
Copy the `preprocess_qwen` function from the `preprocess_v1` function and modify the following:
```
round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # instruction_len is before the answer
```

Add a new conversation template such as `conv_v1_qwen` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_v1_qwen = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1_qwen",  # replace
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",  # replace with eos_token
)
```

Don't forget to register the newly defined conversation template [here]().

```python
conv_templates = {
    ...
    "v1_qwen": conv_v1_qwen,  # the key is "v1_qwen"
    ...
}
```

Remember the key for the registered dialogue conversation, such as `v1_qwen`. And modify the `--version v1_qwen` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**

### PhiTokenizer
Insert the following code after initializing the tokenizer:
```python
tokenizer.add_special_tokens({
    'unk_token': '<|extra_0|>',
#    'eos_token': '<|endoftext|>'  Not needed because it already exists.
})
```
Copy the `preprocess_phi` function from the `preprocess_v1` function and modify the following:
```
round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # instruction_len is before the answer
```

Add a new conversation template such as `conv_v1_phi` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_v1_phi = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1_phi",  # replace
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",  # replace with eos_token
)
```

Don't forget to register the newly defined conversation template [here]().

```python
conv_templates = {
    ...
    "v1_phi": conv_v1_phi,  # the key is "v1_phi"
    ...
}
```

Remember the key for the registered dialogue conversation, such as `v1_phi`. And modify the `--version v1_phi` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**


## Don't have special tokens, but can NOT add special tokens

For those tokenizers that don't have special tokens, but can add special tokens, such as StableLMTokenizer. You need to make sure the tokenizer has `unk_token`. Generally, this type of tokenizer will have `pad_token`.

```
tokenizer.unk_token = tokenizer.pad_token 
```
Copy the `preprocess_stablelm` function from the `preprocess_v1` function and modify the following:
```
total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # pad_token_id == eos_token_id
...
round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # instruction_len is before the answer
```

Add a new conversation template such as `conv_v1_stablelm` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_v1_stablelm = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1_stablelm",  # replace
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",  # replace with eos_token
)
```

Don't forget to register the newly defined conversation template [here]().

```python
conv_templates = {
    ...
    "v1_stablelm": conv_v1_stablelm,  # the key is "v1_stablelm"
    ...
}
```

Remember the key for the registered dialogue conversation, such as `v1_stablelm`. And modify the `--version v1_stablelm` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**

## The behavior of the tokenizer is consistent with `LlamaTokenizer`

### LlamaTokenizer
If the behavior of your tokenizer is consistent with `LlamaTokenizer`. You can just use the already defined conversation template.

For example, for the `LlamaTokenizer`, `bos_token` is `<s>`, `eos_token` is `</s>`, and `unk_token` is `<unk>`.
When the tokenizer encodes one sentence, the resulting output should include the `bos_token_id`. In following example, the `bos_token_id` is 1.


```python
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", cache_dir='cache_dir')
tokenizer(['This is first sentence', 'Test'], return_tensors='pt', padding=True)
# Output: {'input_ids': tensor([[    1,   910,   338,   937, 10541],
#                               [    1,  4321,     0,     0,     0]]),
#          'attention_mask': tensor([[1, 1, 1, 1, 1],
#                                    [1, 1, 0, 0, 0]])}
```
Passing the `--version v1` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**

## Converting models to MoE architectures

Refer to [llava_qwen_moe.py](moellava/model/language_model/llava_llama_moe.py) and [llava_llama_moe.py](moellava/model/language_model/llava_llama_moe.py)

