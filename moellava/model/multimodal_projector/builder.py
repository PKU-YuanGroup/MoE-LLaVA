import torch
import torch.nn as nn
import re

from einops import rearrange

from moellava.model.multimodal_projector.pool_block import Pool_Block
from moellava.model.multimodal_projector.qformer import qformer_config_template, Blip2Model, cheap_qformer_config_template, \
    Cheap_Blip2Model
from moellava.model.multimodal_projector.simple_block import SimpleBlock, Cheap_SimpleBlock


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}



def build_image_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'image_projector_type', 'linear')

    is_cheap = 'cheap' in projector_type
    projector_type = projector_type.replace('cheap_', '') if is_cheap else projector_type

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('qformer'):  # qformer4_36
        qformer_config = cheap_qformer_config_template(config, projector_type) if is_cheap else qformer_config_template(config, projector_type)
        return Cheap_Blip2Model(qformer_config) if is_cheap else Blip2Model(qformer_config)

    elif projector_type.startswith('simple'):  # simple_in0_out0
        pattern = r"simple_in(\d+)_out(\d+)"
        match = re.search(pattern, projector_type)
        num_in_block = int(match.group(1))
        num_out_block = int(match.group(2))
        return Cheap_SimpleBlock(config.mm_hidden_size, config.hidden_size, num_in_block, num_out_block) if is_cheap else SimpleBlock(config.mm_hidden_size, config.hidden_size, num_in_block, num_out_block)

    elif projector_type.startswith('pool'):  # pool_
        projector_type = projector_type.replace('pool_', '')
        return Pool_Block(projector_type, config)

    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')



def build_video_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'video_projector_type', 'linear')

    is_cheap = 'cheap' in projector_type
    projector_type = projector_type.replace('cheap_', '') if is_cheap else projector_type

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('qformer'):  # qformer4_36
        qformer_config = cheap_qformer_config_template(config, projector_type) if is_cheap else qformer_config_template(config, projector_type)
        return Cheap_Blip2Model(qformer_config) if is_cheap else Blip2Model(qformer_config)

    elif projector_type.startswith('simple'):  # simple_in0_out0
        pattern = r"simple_in(\d+)_out(\d+)"
        match = re.search(pattern, projector_type)
        num_in_block = int(match.group(1))
        num_out_block = int(match.group(2))
        return Cheap_SimpleBlock(config.mm_hidden_size, config.hidden_size, num_in_block, num_out_block) if is_cheap else SimpleBlock(config.mm_hidden_size, config.hidden_size, num_in_block, num_out_block)

    elif projector_type.startswith('pool'):  # pool_
        projector_type = projector_type.replace('pool_', '')
        return Pool_Block(projector_type, config)

    else:
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class MLP(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
    def forward(self, x):
        return self.mlp(x)

class build_projector(nn.Module):
    def __init__(self, config, delay_load=False, **kwargs):
        super(build_projector, self).__init__()
        mm_image_tower = getattr(config, 'mm_image_tower', None)
        mm_video_tower = getattr(config, 'mm_video_tower', None)
        self.image_spatial_proj = build_image_projector(config, delay_load=False, **kwargs) if mm_image_tower is not None else None

        if mm_video_tower is not None:
            self.video_patch_proj = build_video_projector(config, delay_load=False, **kwargs)
            self.video_spatial_proj = MLP(config.mm_hidden_size, config.hidden_size) if config.video_spatial_proj else None
            self.video_temproal_proj = MLP(config.mm_hidden_size, config.hidden_size) if config.video_temproal_proj else None
            self.video_global_proj = MLP(config.mm_hidden_size, config.hidden_size) if config.video_global_proj else None

        else:
            self.video_patch_proj = nn.Identity()
            self.video_spatial_proj = nn.Identity()
            self.video_temproal_proj = nn.Identity()
            self.video_global_proj = nn.Identity()


    def forward_image(self, image_feature):
        return self.image_spatial_proj(image_feature)

    def forward_video(self, video_feature):
        global_feature, origin_patch_feature = video_feature[:, :, 0, :], video_feature[:, :, 1:, :]  # [b, t, c], [b, t, n, c]
        b, t, n, c = origin_patch_feature.shape

        # print(video_feature.shape, origin_patch_feature.shape)
        patch_feature = self.video_patch_proj(rearrange(origin_patch_feature, 'b t n c -> (b t) n c'))  # [b, t, n, c] -> [bt, new_n, c]
        patch_feature = rearrange(patch_feature, '(b t) new_n c -> b t new_n c', b=b)  # [bt, new_n, c] -> [b, t, new_n, c]

        video_hidden_state = patch_feature

        if self.video_temproal_proj:
            temproal_feature = self.video_temproal_proj(origin_patch_feature.mean(2)).unsqueeze(2)  # [b, t, n, c] -> [b, t, 1, c]
            video_hidden_state = torch.cat([video_hidden_state, temproal_feature], dim=2)

        if self.video_global_proj:
            global_feature = self.video_global_proj(global_feature).unsqueeze(2)  # [b, t, c] -> [b, t, 1, c]
            video_hidden_state = torch.cat([global_feature, video_hidden_state], dim=2)

        if self.video_spatial_proj:
            spatial_feature = self.video_spatial_proj(origin_patch_feature.mean(1))  # [b, t, n, c] -> [b, n, c]

        video_hidden_state_list = []
        for i in range(b):
            tmp = []
            for j in range(t):
                if j+1 != t:
                    tmp.append(video_hidden_state[i][j])  # 1+1+new_n, c
                elif self.video_spatial_proj:  # add to tail
                    tmp.append(torch.cat([video_hidden_state[i][j], spatial_feature[i]], dim=0))  # 1+1+new_n+n, c
                else:
                    tmp.append(video_hidden_state[i][j])  # 1+1+new_n, c
            video_hidden_state_list.append(tmp)

        # video_hidden_state_list = []
        # for i in range(b):
        #     for j in range(t):
        #         if j+1 != t:
        #             video_hidden_state_list.append(video_hidden_state[i][j])  # 1+1+new_n, c
        #         elif self.video_spatial_proj:  # add to tail
        #             video_hidden_state_list.append(torch.cat([video_hidden_state[i][j], spatial_feature[i]], dim=0))  # 1+1+new_n+n, c
        #         else:
        #             video_hidden_state_list.append(video_hidden_state[i][j])  # 1+1+new_n, c


        return video_hidden_state_list
    # def forward(self, x):
    #     if x.ndim == 3:  # batch consists of images, [b, n, c]
    #         return self.forward_image(x)
    #     elif x.ndim == 4:  # batch consists of videos, [b, t, 1+n, c]
    #         return self.forward_video(x)
    #     else:
    #         raise NotImplementedError(f'We do not know the shape of {x.shape}')
