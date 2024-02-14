#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel, _import_flash_attn, SUPPORT_BF16, SUPPORT_FP16, \
    SUPPORT_CUDA, logger
from .qwen.configuration_qwen import QWenConfig

from transformers.modeling_outputs import CausalLMOutputWithPast

from .qwen.tokenization_qwen import QWenTokenizer
from ..llava_arch import LlavaMetaModel, LlavaQWenMetaForCausalLM
import torch.distributed as dist


class LlavaQWenConfig(QWenConfig):
    model_type = "llava_qwen"


class LlavaQWenModel(LlavaMetaModel, QWenModel):
    config_class = LlavaQWenConfig

    def __init__(self, config: QWenConfig):
        super(LlavaQWenModel, self).__init__(config)

    def embed_tokens(self, input_ids):
        return self.wte(input_ids)

class LlavaQWenForCausalLM(QWenLMHeadModel, LlavaQWenMetaForCausalLM):
    config_class = LlavaQWenConfig

    def __init__(self, config):
        super(QWenLMHeadModel, self).__init__(config)
        # import ipdb
        # ipdb.set_trace()
        assert (
                config.bf16 + config.fp16 + config.fp32 <= 1
        ), "Only one of \"bf16\", \"fp16\", \"fp32\" can be true"

        # autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0
        autoset_precision = True

        if autoset_precision:
            if SUPPORT_BF16:
                logger.warn(
                    "The model is automatically converting to bf16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.bf16 = True
            elif SUPPORT_FP16:
                logger.warn(
                    "The model is automatically converting to fp16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.fp16 = True
            else:
                config.fp32 = True

        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn(
                "Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\".")
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn(
                "Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster")
        if config.fp32:
            if SUPPORT_BF16:
                logger.warn(
                    "Your device support faster inference by passing bf16=True in \"AutoModelForCausalLM.from_pretrained\".")
            elif SUPPORT_FP16:
                logger.warn(
                    "Your device support faster inference by passing fp16=True in \"AutoModelForCausalLM.from_pretrained\".")

        if config.use_flash_attn == "auto":
            # if config.bf16 or config.fp16:
            if config.bf16:
                logger.warn("Try importing flash-attention for faster inference...")
                config.use_flash_attn = True
            else:
                config.use_flash_attn = False
        config.use_flash_attn = False
        if config.use_flash_attn and config.fp32:
            logger.warn("Flash attention will be disabled because it does NOT support fp32.")

        if config.use_flash_attn:
            _import_flash_attn()

        self.transformer = LlavaQWenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
        self.post_init()

    def get_model(self):
        return self.transformer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # import ipdb
        # ipdb.set_trace()
        # print(f'rank {dist.get_rank()}', 'before prepare_inputs_labels_for_multimodal')
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after prepare_inputs_labels_for_multimodal')

        out = super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )
        # dist.barrier()
        # print(f'rank {dist.get_rank()}', 'after LLM')
        return out

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # import ipdb
        # ipdb.set_trace()
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("llava_qwen", LlavaQWenConfig)
AutoTokenizer.register(LlavaQWenConfig, QWenTokenizer)
AutoModelForCausalLM.register(LlavaQWenConfig, LlavaQWenForCausalLM)
