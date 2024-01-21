## Data preparation

- The LLaVA-PT is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- The Hybird-FT is from [SViT](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning), [LVIS](https://github.com/X2FD/LVIS-INSTRUCT4V), [LRV](https://github.com/FuxiaoLiu/LRV-Instruction), [MIMIC-IT](https://github.com/Luodian/Otter).
- The LLaVA-FT is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- Download the training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1rwub9o0T3_7ZHbPZzCiLZw?pwd=0yhi), [Google Disk](https://drive.google.com/file/d/13YxtVowfhUIpGOCODhKFstoRBvogF4od/view?usp=sharing) or [Peking University Disk]()


We also provide the processed data as follows.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Data group</th><th>Usage</th><th>Link</th>
    </tr>
    <tr align="center">
        <td>LLaVA-PT</td><td>Stage 1</td><td>LLaVA 1.5-558k (<a href="https://pan.baidu.com/s/1UZiRORpXwAHdKPgrUi1nDA?pwd=7xgx">BaiDu</a>)</td>
    </tr>
    <tr align="center">
        <td>Hybird-FT</td><td>Stage 2</td><td>SViT-157k (<a href="https://pan.baidu.com/s/1PtcTck4xC0fAE0QS0OYc8Q?pwd=ko9x">BaiDu</a>), LVIS-220k (<a href="https://pan.baidu.com/s/1-MWrPGZptFFBO1_4tniAXA?pwd=ivxg">BaiDu</a>), LRV-331k (<a href="https://pan.baidu.com/s/1sYnfRN_yFuo719fNA_BV_w?pwd=lmai">BaiDu</a>), MIMIC-IT-256k (<a href="">BaiDu</a>)</td>
    </tr>
    <tr align="center">
        <td>LLaVA-FT</td><td>Stage 3</td><td>LLaVA 1.5-mix-665k (<a href="https://pan.baidu.com/s/1xC9E6VuOOEBV5iieve0Z7A?pwd=2o0a">BaiDu</a>)</td>
    </tr>
</table>
</div>

After downloading all of them, organize the data as follows in ```DATA_ROOT```. 

```Shell
DATA_ROOT
├── llava_image
├── llava_image_tune
├── lvis_tune
├── lrv_tune
├── svit_tune
└── mimicit_tune
    └── LA
```


## Training
Specify your `DATA_ROOT` according to the data preparation.
- Stage 1 pretraining script: [pretrain.sh](scripts/v1_5/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](scripts/v1_5/finetune.sh).
Make sure you understand the behavior of the tokenizer.
- Stage 3 moe-tuning script: [moe-finetune.sh]().
