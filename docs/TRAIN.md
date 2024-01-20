## Data preparation

- The images pretraining dataset is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- The images tuning dataset is from [LLaVA](https://github.com/haotian-liu/LLaVA).
- Download the training annotations. You can download from [Baidu Disk](https://pan.baidu.com/s/1BipI3_f--GRWqaWTGYp-Jg?pwd=wkl0), [Google Disk](https://drive.google.com/file/d/11-1NBXNeiNQE2wPbue1dFph_Na_EHRYG/view?usp=drive_link) or [Peking University Disk](https://disk.pku.edu.cn:443/link/84783AB54553DFA150C1C5E82C16EB29)


We also provide the processed data as follows.
<div align="center">
<table border="1" width="100%">
    <tr align="center">
        <th>Datasets</th><th>Baidu Disk</th>
    </tr>
    <tr align="center">
        <td>Image pretraining</td><td><a href="https://pan.baidu.com/s/17GYcE69FcJjjUM0e4Gad2w?pwd=9ga3">Link</a></td>
    </tr>
    </tr>
    <tr align="center">
        <td>Image tuning</td><td><a href="https://pan.baidu.com/s/1l-jT6t_DlN5DTklwArsqGw?pwd=o6ko">Link</a></td>
    </tr>
</table>
</div>

After downloading all of them, organize the data as follows in ```DATA_ROOT```. 

```Shell
DATA_ROOT
├── llava_image
└── llava_image_tune
```


## Training
Specify your `DATA_ROOT` according to the data preparation.
- Stage 1 pretraining script: [pretrain.sh](scripts/v1_5/pretrain.sh). 
- Stage 2 tuning script: [finetune.sh](scripts/v1_5/finetune.sh).
