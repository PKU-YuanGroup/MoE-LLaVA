

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

## Modeling



