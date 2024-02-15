import argparse

import torch
from peft.tuners.lora import LoraLayer
from peft.utils import ModulesToSaveWrapper, _get_submodules
from torch import nn
from transformers.pytorch_utils import Conv1D

from moellava.model.builder import load_pretrained_model
from moellava.mm_utils import get_model_name_from_path


def _replace_module(parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias"):
        if old_module.bias is not None:
            new_module.bias = old_module.bias

    if getattr(old_module, "state", None) is not None:
        new_module.state = old_module.state
        new_module.to(old_module.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora_" in name:
            module.to(old_module.weight.device)
        if "ranknum" in name:
            module.to(old_module.weight.device)

def _unload_and_optionally_merge(model, merge=True):
    if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue
        if isinstance(target, LoraLayer):
            if isinstance(target, nn.Embedding):
                new_module = torch.nn.Embedding(target.in_features, target.out_features)
            elif isinstance(target, nn.Conv2d):
                new_module = torch.nn.Conv2d(
                    target.in_channels,
                    target.out_channels,
                    kernel_size=target.kernel_size,
                    stride=target.stride,
                    padding=target.padding,
                    dilation=target.dilation,
                )
            else:
                bias = getattr(target, 'bias', None) is not None
                if getattr(target, "is_target_conv_1d_layer", False):
                    new_module = Conv1D(target.out_features, target.in_features)
                else:
                    new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
            if merge:
                target.merge()
            _replace_module(parent, target_name, new_module, target)

        # save any additional trainable modules part of `modules_to_save`
        if isinstance(target, ModulesToSaveWrapper):
            setattr(parent, target_name, target.modules_to_save[target.active_adapter])

    return model


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                           device_map='cpu', merge=True)

    model = model.cuda()
    model_merge = _unload_and_optionally_merge(model)
    model_merge.config.lora_merge = model_merge.config.lora
    delattr(model_merge.config, 'lora')
    # import ipdb
    # ipdb.set_trace()
    model_merge.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None, required=False)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--local_rank", default=-1, type=int, required=False)

    args = parser.parse_args()

    merge_lora(args)
