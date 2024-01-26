

- The most **IMPORTANT** thing, make sure you understand the behavior of the tokenizer. 
- We provide some samples on how different tokenizer behaviors should be changed. 
- At the end it describes how to convert LLaVA style models to the MoE architecture.

## Don't have special tokens, but can add special tokens

For those tokenizers that don't have special tokens, but can add special tokens, such as QWenTokenizer or PhiTokenizer. You need to add special tokens.

### QWenTokenizer

#### Tokenizer

Insert the following code after initializing the tokenizer [here]():
```python
tokenizer.add_special_tokens({
    'unk_token': '<|extra_0|>',
    'eos_token': '<|endoftext|>'
})
```

#### `preprocess_qwen` function

Copy the `preprocess_qwen` function from the `preprocess_v1` function and modify the following:
```
round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # instruction_len is before the answer
```

Defining the use of `preprocess_qwen` in the `preprocess` function [here]().
```
if conversation_lib.default_conversation.version.startswith("qwen"):  # for qwen
    return preprocess_qwen(sources, tokenizer, has_image=has_image)
```

#### `conv_qwen` conversation template

Add a new conversation template such as `conv_qwen` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_qwen = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="qwen",  # replace
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
    "qwen": conv_qwen,  # the key is "qwen"
    ...
}
```

Remember the key for the registered dialogue conversation, such as `qwen`. And modify the `--version qwen` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**

### PhiTokenizer

#### Tokenizer

Insert the following code after initializing the tokenizer [here]():
```python
tokenizer.add_special_tokens({
    'unk_token': '<|extra_0|>',
#    'eos_token': '<|endoftext|>'  Not needed because it already exists.
})
```

#### `preprocess_phi` function

Copy the `preprocess_phi` function from the `preprocess_v1` function and modify the following:
```
round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # instruction_len is before the answer
```

Defining the use of `preprocess_phi` in the `preprocess` function [here]().
```
if conversation_lib.default_conversation.version.startswith("phi"):  # for phi
    return preprocess_phi(sources, tokenizer, has_image=has_image)
```

#### `conv_phi` conversation template

Add a new conversation template such as `conv_phi` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_phi = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi",  # replace
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
    "phi": conv_phi,  # the key is "phi"
    ...
}
```

Remember the key for the registered dialogue conversation, such as `phi`. And modify the `--version phi` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**


## CAN NOT add special tokens

### StableLMTokenizer

#### Tokenizer

For those tokenizers that can **not** add special tokens, such as `StableLMTokenizer`.

First find all the special tokens of the tokenizer.

```
tokenizer.special_tokens
>>> {'<|endoftext|>': 100257, '<|fim_prefix|>': 100258, '<|fim_middle|>': 100259, '<|fim_suffix|>': 100260, '<|fim_pad|>': 100261, '<gh_stars>': 100262, '<filename>':  100263, '<issue_start>': 100264, '<issue_comment>': 100265, '<issue_closed>': 100266, '<jupyter_start>': 100267, '<jupyter_text>': 100268, '<jupyter_code>': 100269, '<jupyter_output>': 100270, '<empty_output>': 100271, '<commit_before>': 100272, '<commit_msg>': 100273, '<commit_after>': 100274, '<reponame>': 100275, '<|endofprompt|>': 100276, '<|im_start|>': 100277, '<|im_end|>': 100278, '<|pause|>': 100279, '<|reg0|>': 100280, '<|reg1|>': 100281, '<|reg2|>': 100282, '<|reg3|>': 100283, '<|reg4|>': 100284, '<|reg5|>': 100285, '<|reg6|>': 100286, '<|reg7|>': 100287, '<|extra0|>': 100288}
```

Choosing a less important token, e.g., `<|reg0|>`.  You need to make sure the tokenizer has `unk_token` [here]().

```
tokenizer.unk_token = '<|reg0|>'
```

#### `preprocess_stablelm` function

Copy the `preprocess_stablelm` function from the `preprocess_v1` function and modify the following:
```
total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # pad_token_id == eos_token_id
...
round_len = len(tokenizer_image_token(rou, tokenizer)) + 1  # for eos_token
instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # instruction_len is before the answer
```

Defining the use of `preprocess_stablelm` in the `preprocess` function [here]().
```
if conversation_lib.default_conversation.version.startswith("stablelm"):  # for stablelm
    return preprocess_stablelm(sources, tokenizer, has_image=has_image)
```

#### `conv_stablelm` conversation template

Add a new conversation template such as `conv_stablelm` [here](), replacing `sep2` with `eos_token`, and modify the value of `version`.

```python
conv_stablelm = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="stablelm",  # replace
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
    "stablelm": conv_stablelm,  # the key is "stablelm"
    ...
}
```

Remember the key for the registered dialogue conversation, such as `stablelm`. And modify the `--version stablelm` in the commands for Stage 2 and Stage 3. **DO NOT need to modify the `--version plain` in Stage 1.**

## The behavior of the tokenizer is consistent with `LlamaTokenizer`

### LlamaTokenizer

If the behavior of your tokenizer is consistent with `LlamaTokenizer`. You can just use the already defined conversation template. Beware of the differences brought about by different transformers versions, **we strongly recommend using `LlamaTokenizer` on version 4.31.0**.

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

Refer to [llava_stablelm_moe.py](moellava/model/language_model/llava_stablelm_moe.py), [llava_qwen_moe.py](moellava/model/language_model/llava_llama_moe.py), [llava_phi_moe.py](moellava/model/language_model/llava_phi_moe.py), [llava_mistral_moe.py](moellava/model/language_model/llava_mistral_moe.py) and [llava_llama_moe.py](moellava/model/language_model/llava_llama_moe.py)

