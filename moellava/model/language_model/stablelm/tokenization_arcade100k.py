# coding=utf-8
# Copyright (c) 2023 Alibaba Cloud & Stability AI.
#
# Tongyi Qianwen LICENSE AGREEMENT:
# https://github.com/QwenLM/Qwen/blob/5aa84bdfd3237b37f01bc88cd49b3279b9a71d0b/Tongyi%20Qianwen%20LICENSE%20AGREEMENT
"""Tokenization classes for Arcade100k."""

import base64
import os
import unicodedata
from typing import Collection, Dict, List, Set, Tuple, Union

import tiktoken
from transformers.utils import logging
from transformers import PreTrainedTokenizer, AddedToken

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "arcade100k.tiktoken"}
NAME = "arcade100k"


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in contents.splitlines() if line)
    }


ENDOFTEXT = "<|endoftext|>"
FIM = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
]
# `StarCoder` Tokens
CODE = [
    "<gh_stars>",
    "<filename>",
    "<issue_start>",
    "<issue_comment>",
    "<issue_closed>",
    "<jupyter_start>",
    "<jupyter_text>",
    "<jupyter_code>",
    "<jupyter_output>",
    "<empty_output>",
    "<commit_before>",
    "<commit_msg>",
    "<commit_after>",
    "<reponame>",
]
CHAT = [
    "<|im_start|>",  # Chat: Input message start
    "<|im_end|>",  # Chat: Input message end
]
PAUSE = "<|pause|>"  # Think before you speak (https://arxiv.org/abs/2310.02226)
REGISTERS = [
    f"<|reg{i}|>" for i in range(0, 8)
]  # Register 0 sink token (https://arxiv.org/abs/2309.17453)
ENDOFPROMPT = "<|endofprompt|>"
SPECIAL_TOKENS_NAMES = (
    [ENDOFTEXT]
    + FIM
    + CODE
    + [ENDOFPROMPT]
    + CHAT
    + [PAUSE]
    + REGISTERS
    + ["<|extra0|>"]
)
START_ID = 100257
SPECIAL_TOKENS = {t: START_ID + i for i, t in enumerate(SPECIAL_TOKENS_NAMES)}


def _arcade100k(vocab_file: str):
    mergeable_ranks = _load_tiktoken_bpe(vocab_file)

    return {
        "name": NAME,
        "pat_str": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": SPECIAL_TOKENS,
    }


class Arcade100kTokenizer(PreTrainedTokenizer):
    """
    Construct a Arcade100k tokenizer backed by `tiktoken`.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        errors (`str`, *optional*, defaults to `"replace"`):
            How to handle errors in decoding UTF-8 byte sequences.
            WARNING: the default behaviour of this function is lossy, since decoded bytes are not
            guaranteed to be valid UTF-8. You can control this behaviour using the `errors` parameter,
            for instance, setting `errors=strict`.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        errors: str = "replace",
        **kwargs,
    ):
        super().__init__(errors=errors, **kwargs)
        self._tiktoken_config = _arcade100k(vocab_file)
        self.tokenizer = tiktoken.Encoding(**self._tiktoken_config)

        # TODO: Remove this assertion
        assert (
            len(self.tokenizer._mergeable_ranks)
            + len(self.tokenizer._special_tokens)
            + 1
            == self.tokenizer.n_vocab
        ), f"{len(self.tokenizer._mergeable_ranks) + len(self.tokenizer._special_tokens)} != {self.tokenizer.n_vocab} in encoding"

        self.decoder = {i: n for n, i in self.tokenizer._mergeable_ranks.items()}
        self.decoder.update({i: n for n, i in self.tokenizer._special_tokens.items()})
        self.eos_token = self.decoder[self.tokenizer.eot_token]
        self.pad_token = self.decoder[self.tokenizer.eot_token]
        # Expose for convenience
        self.mergeable_ranks = self.tokenizer._mergeable_ranks
        self.special_tokens = self.tokenizer._special_tokens

    def __len__(self):
        return self.tokenizer.n_vocab

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def get_vocab(self) -> Dict[bytes, int]:
        return self.tokenizer._mergeable_ranks

    def convert_tokens_to_ids(
        self, tokens: Union[bytes, str, List[Union[bytes, str]]]
    ) -> List[int]:
        ids = []
        if isinstance(tokens, (str, bytes)):
            if tokens in self.tokenizer._special_tokens:
                return self.tokenizer._special_tokens[tokens]
            else:
                return self.tokenizer._mergeable_ranks.get(tokens)
        for token in tokens:
            if token in self.tokenizer._special_tokens:
                ids.append(self.tokenizer._special_tokens[token])
            else:
                ids.append(self.tokenizer._mergeable_ranks.get(token))
        return ids

    def _add_tokens(
        self,
        new_tokens: Union[List[str], List[AddedToken]],
        special_tokens: bool = False,
    ) -> int:
        if not special_tokens and new_tokens:
            raise ValueError("Adding regular tokens is not supported")
        for token in new_tokens:
            surface_form = token.content if isinstance(token, AddedToken) else token
            if surface_form not in SPECIAL_TOKENS:
                raise ValueError("Adding unknown special tokens is not supported")
        return 0

    def save_vocabulary(self, save_directory: str, **kwargs) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary).

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        file_path = os.path.join(save_directory, "arcade100k.tiktoken")
        with open(file_path, "w", encoding="utf8") as w:
            for k, v in self.tokenizer._mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + " " + str(v) + "\n"
                w.write(line)
        return (file_path,)

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
        **kwargs,
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        ):
            tokens.append(self.decoder[t])
        return tokens

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.tokenizer._special_tokens:
            return self.tokenizer._special_tokens[token]
        if token in self.tokenizer._mergeable_ranks:
            return self.tokenizer._mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.tokenizer.eot_token]
        return self.tokenizer.decode(token_ids)