import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from .intern_vit_model.modeling_intern_vit import InternVisionModel 
from .intern_vit_model.configuration_intern_vit import InternVisionConfig

class InternViTVisionTower:
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        pass
