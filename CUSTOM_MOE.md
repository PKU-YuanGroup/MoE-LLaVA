
# We provide some samples on how different tokenizer behaviors should be changed to fit MoE-LLaVA.

## QWenTokenizer
QWenTokenizer does not have special tokens, which one needs to define.
You may need to insert the following code after initializing the tokenizer:
```python
tokenizer.add_special_tokens({
    'unk_token': '<|extra_0|>',
    'bos_token': '<|extra_1|>',
    'eos_token': '<|endoftext|>'
})
tokenizer.pad_token = tokenizer.unk_token  # just for padding demo
```
This is what we do with the `QWenTokenizer`. After add special tokens to `QWenTokenizer`, `bos_token_id = 151647` and `pad_token_id = 151646`:
```python
{
    'input_ids': tensor([[151647,   1986,    374,   1156,  11652], [151647,   2271, 151646, 151646, 151646]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
}
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

## PhiTokenizer


Add a new conversation template such as `conv_v0_phi` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_v0_phi = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v0_phi",  # replace
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
    "v0_phi": conv_v0_phi,  # the key is "v0_phi"
    ...
}
```


Due to `pad_token_id == eos_token_id` in phi-2, so we define a new process function `preprocess_v0` [here]()

Remember the key for the registered dialogue conversation, such as `v0_phi`. And modify the `--version v0_phi` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**


## LlamaTokenizer
If your tokenizer is from llama, then make sure your tokenizer behavior is consistent with llama.

For example, for the `LlamaTokenizer`, `bos_token` is `<s>`, `eos_token` is `</s>`, and `unk_token` is `<unk>`.
When the tokenizer encodes one sentence, the resulting output should include the `bos_token_id`. In following example, the `bos_token_id` is 1.


```python
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", cache_dir='cache_dir')
tokenizer(['This is first sentence', 'Test'], return_tensors='pt', padding=True)
tokenizer.pad_token = tokenizer.unk_token  # just for padding demo
# Output: {'input_ids': tensor([[    1,   910,   338,   937, 10541],
#                               [    1,  4321,     0,     0,     0]]),
#          'attention_mask': tensor([[1, 1, 1, 1, 1],
#                                    [1, 1, 0, 0, 0]])}
```
Passing the `--version v1` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**

## Modeling

Refer to [llava_qwen_moe.py](moellava/model/language_model/llava_llama_moe.py) and [llava_llama_moe.py](moellava/model/language_model/llava_llama_moe.py)

