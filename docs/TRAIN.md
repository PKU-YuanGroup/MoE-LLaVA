## Data preparation

- The LLaVA-PT is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- The Hybird-FT is from [SViT](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning), [LVIS](https://github.com/X2FD/LVIS-INSTRUCT4V), [LRV](https://github.com/FuxiaoLiu/LRV-Instruction), [MIMIC-IT](https://github.com/Luodian/Otter).
- The LLaVA-FT is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- Download the training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1rwub9o0T3_7ZHbPZzCiLZw?pwd=0yhi), [Google Disk](https://drive.google.com/file/d/13YxtVowfhUIpGOCODhKFstoRBvogF4od/view?usp=sharing), [Peking University Disk](https://disk.pku.edu.cn/link/AA10683317FB824FB9B2427A6B268EAADB) or [Hugging Face](https://huggingface.co/datasets/LanguageBind/MoE-LLaVA/tree/main/train_json)


We also provide the processed data as follows. The link is to BaiDu Disk.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Data group</th><th>Usage</th><th>Link</th>
    </tr>
    <tr align="center">
        <td>LLaVA-PT</td><td>Stage 1</td><td><a href="https://pan.baidu.com/s/1UZiRORpXwAHdKPgrUi1nDA?pwd=7xgx">LLaVA 1.5-558k</a></td>
    </tr>
    <tr align="center">
        <td>Hybird-FT</td><td>Stage 2</td><td><a href="https://pan.baidu.com/s/1PtcTck4xC0fAE0QS0OYc8Q?pwd=ko9x">SViT-157k</a>, <a href="https://pan.baidu.com/s/1-MWrPGZptFFBO1_4tniAXA?pwd=ivxg">LVIS-220k</a>, <a href="https://pan.baidu.com/s/1sYnfRN_yFuo719fNA_BV_w?pwd=lmai">LRV-331k</a>, <a href="https://pan.baidu.com/s/1w0Wr8d-IhIUuRyKbuoPwyw?pwd=4big">MIMIC-IT-256k</a></td>
    </tr>
    <tr align="center">
        <td>LLaVA-FT</td><td>Stage 3</td><td><a href="https://pan.baidu.com/s/1xC9E6VuOOEBV5iieve0Z7A?pwd=2o0a">LLaVA 1.5-mix-665k</a></td>
    </tr>
</table>
</div>

**For those who can not easily access to BaiDu Disk**, you can download data from [Hugging Face](https://huggingface.co/datasets/LanguageBind/MoE-LLaVA).

After downloading all of them, organize the data as follows in ```IMAGE_FOLDER```. 

```Shell
IMAGE_FOLDER
├── llava_image
├── llava_image_tune
├── lvis_tune
├── lrv_tune
├── svit_tune
└── mimicit_tune
    └── LA
```


## Training
Specify your `IMAGE_FOLDER` and `JSON_FOLDER` according to the data preparation.

For training on 384 resolution, we use `google/siglip-so400m-patch14-384` as `image_tower`. Notably, if you pass the `--image_tower google/siglip-so400m-patch14-384`, you should upgrade the version of transformers to 4.37.0.

### Qwen
- Stage 1 pretraining script: [pretrain.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/qwen/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/qwen/finetune.sh).
- Stage 3 moe-tuning script: [finetune_moe.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/qwen/finetune_moe.sh).
  
### Phi2
- Stage 1 pretraining script: [pretrain.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/phi2/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/phi2/finetune.sh).
- Stage 3 moe-tuning script: [finetune_moe.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/phi2/finetune_moe.sh).
  
### StableLM
- Stage 1 pretraining script: [pretrain.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/stablelm/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/stablelm/finetune.sh).
- Stage 3 moe-tuning script: [finetune_moe.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/stablelm/finetune_moe.sh).
  
### OpenChat

<!-- OpenChat seems to have bugs in `transformer==4.36.2`. Please `pip install transformers==4.34.0`. -->

- Stage 1 pretraining script: [pretrain.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/openchat/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/openchat/finetune.sh).
- Stage 3 moe-tuning script: [finetune_moe.sh](https://github.com/PKU-YuanGroup/MoE-LLaVA/tree/main/scripts/v1/openchat/finetune_moe.sh).
