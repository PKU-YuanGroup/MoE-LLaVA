

## Tokenizer

Make sure your tokenizer has `bos_token`, `eos_token`, `unk_token`, and they are all different. 

For example, for the `LlamaTokenizer`, `bos_token` is `<s>`, `eos_token` is `</s>`, and `unk_token` is `<unk>`.
When the tokenizer encodes one sentence, the resulting output should include the `bos_token_id`. In following example, the `bos_token_id` is 1.


```python
tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", cache_dir='cache_dir')
tokenizer(['This is first sentence', 'Test'], return_tensors='pt', padding=True)
tokenizer.pad_token = tokenizer.unk_token  # just for demo
# Output: {'input_ids': tensor([[    1,   910,   338,   937, 10541],
#                               [    1,  4321,     0,     0,     0]]),
#          'attention_mask': tensor([[1, 1, 1, 1, 1],
#                                    [1, 1, 0, 0, 0]])}
```

If your tokenizer doesn't have these special tokens **(if they already exist, skip these steps)**, you may need to insert the following code after initializing the tokenizer:
```python
tokenizer.add_special_tokens({
    'unk_token': '<|extra_0|>',
    'bos_token': '<|extra_1|>',
    'eos_token': '<|endoftext|>'
})
tokenizer.pad_token = tokenizer.unk_token  # just for demo
```
This is what we do with the `QWenTokenizer`. After add special tokens to `QWenTokenizer`:
```python
{
    'input_ids': tensor([[151647,   1986,    374,   1156,  11652], [151647,   2271, 151646, 151646, 151646]]),
    'token_type_ids': tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
}
```
## Conversation

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
    sep2="<|endoftext|>",  # replace
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

Remember the key for the registered dialogue conversation, such as `v1_qwen`. And modify the `--version` in the commands for Stage 2 and Stage 3. **No need to modify the `--version` in Stage 1.**

## Modeling



