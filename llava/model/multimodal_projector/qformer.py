import re
from typing import Optional

import torch
from torch import nn
from transformers import Blip2PreTrainedModel, Blip2Config, Blip2QFormerModel, PretrainedConfig


class Blip2Model(Blip2PreTrainedModel):
    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.proj_in = nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(config.hidden_size, config.hidden_size))

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.proj_out = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(config.hidden_size, config.hidden_size))
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(pixel_values.shape)
        pixel_values = self.proj_in(pixel_values)
        image_embeds = pixel_values


        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        # print('qformer out', query_outputs.shape)
        # print(query_outputs.shape)
        query_outputs = self.proj_out(query_outputs)
        return query_outputs


def qformer_config_template(config, projector_type):
    pattern = r"qformer(\d+)_(\d+)"

    match = re.search(pattern, projector_type)
    num_hidden_layers = int(match.group(1))
    num_query_tokens = int(match.group(2))

    qformer_config = type('Blip2Config', (PretrainedConfig,), {
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "model_type": "blip-2",
        "num_query_tokens": num_query_tokens,
        "hidden_size": config.hidden_size,
        "mm_hidden_size": config.mm_hidden_size,
        "qformer_config": type('qformer_config', (PretrainedConfig,), {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": None,
            "cross_attention_frequency": 1,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "encoder_hidden_size": config.hidden_size,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": config.hidden_size,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "initializer_range": 0.02,
            "intermediate_size": int(config.hidden_size * 2.6875),
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
            },
            "layer_norm_eps": 1e-12,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 512,
            "min_length": 0,
            "model_type": "blip_2_qformer",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 32,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": num_hidden_layers,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.27.0.dev0",
            "typical_p": 1.0,
            "use_bfloat16": False,
            "vocab_size": 30522
        })()
    })()
    return qformer_config


class Cheap_Blip2Model(Blip2PreTrainedModel):
    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.proj_in = nn.Sequential(nn.Linear(config.mm_hidden_size, config.mm_hidden_size),
                                  nn.GELU(),
                                  nn.Linear(config.mm_hidden_size, config.mm_hidden_size))

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.proj_out = nn.Sequential(nn.Linear(config.mm_hidden_size, config.hidden_size),
                                  nn.GELU(),
                                  nn.Linear(config.hidden_size, config.hidden_size))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds = self.proj_in(pixel_values)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state

        query_outputs = self.proj_out(query_outputs)
        return query_outputs


def cheap_qformer_config_template(config, projector_type):
    pattern = r"qformer(\d+)_(\d+)"

    match = re.search(pattern, projector_type)
    num_hidden_layers = int(match.group(1))
    num_query_tokens = int(match.group(2))

    qformer_config = type('Blip2Config', (PretrainedConfig,), {
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "model_type": "blip-2",
        "num_query_tokens": num_query_tokens,
        "hidden_size": config.hidden_size,
        "mm_hidden_size": config.mm_hidden_size,
        "qformer_config": type('qformer_config', (PretrainedConfig,), {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": None,
            "cross_attention_frequency": 1,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "encoder_hidden_size": config.mm_hidden_size,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": config.mm_hidden_size,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "initializer_range": 0.02,
            "intermediate_size": int(config.mm_hidden_size * 4),
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
            },
            "layer_norm_eps": 1e-12,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 512,
            "min_length": 0,
            "model_type": "blip_2_qformer",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 32,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": num_hidden_layers,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "prefix": None,
            "problem_type": None,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.27.0.dev0",
            "typical_p": 1.0,
            "use_bfloat16": False,
            "vocab_size": 30522
        })()
    })()
    return qformer_config


if __name__ == '__main__':
    config = type('Args', (), {
        "hidden_size": 4096,
        "mm_hidden_size": 1024
    })()
    projector_type = 'qformer2_64'
    qformer_config = qformer_config_template(config, projector_type)
    qformer = Blip2Model(qformer_config)

    # projector_type = 'cheap_qformer4_36'
    # qformer_config = cheap_qformer_config_template(config, projector_type)
    # qformer = Cheap_Blip2Model(qformer_config)
    # print(qformer)

    x = torch.randn(2, 256, 1024)
    y = qformer(x)
    print(y.shape)
    params_count = sum(p.numel() for p in qformer.parameters() if p.requires_grad)
    print(round(params_count/1000000, 2))
