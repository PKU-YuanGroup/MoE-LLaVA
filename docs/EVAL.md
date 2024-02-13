## Data preparation

- Following LLaVA's instructions. **You MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**.
- It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to `eval`. This also provides a general structure for all datasets.

After downloading all of them, organize the data as follows in `eval`.

```Shell
eval
├── gqa
│   ├── answers
│   ├── data
│   └── llava_gqa_testdev_balanced.jsonl
├── llava-bench-in-the-wild
│   ├── answers
│   ├── answers_gpt4.jsonl
│   ├── bard_0718.jsonl
│   ├── bing_chat_0629.jsonl
│   ├── context.jsonl
│   ├── images
│   ├── questions.jsonl
│   ├── README.md
│   └── reviews
├── mmbench
│   ├── answers
│   ├── answers_upload
│   ├── mmbench_dev_20230712.tsv
│   └── mmbench_dev_en_20231003.tsv
├── MME
│   ├── answers
│   ├── convert_answer_to_mme.py
│   └── llava_mme.jsonl
├── mm-vet
│   ├── answers
│   ├── bard_set.json
│   ├── convert_answers.py
│   ├── images
│   ├── llava-mm-vet.jsonl
│   ├── mm-vet.json
│   └── results
├── pope
│   ├── answers
│   ├── coco
│   ├── llava_pope_test.jsonl
│   └── val2014
├── scienceqa
│   ├── answers
│   ├── images
│   ├── llava_test_CQM-A.json
│   ├── pid_splits.json
│   └── problems.json
├── seed_bench
│   ├── answers
│   ├── answers_upload
│   ├── extract_video_frames.py
│   └── llava-seed-bench.jsonl
├── textvqa
│   ├── answers
│   ├── llava_textvqa_val_v051_ocr.jsonl
│   ├── TextVQA_0.5.1_val.json
│   └── train_images
├── vizwiz
│   ├── answers
│   ├── answers_upload
│   ├── llava_test.jsonl
│   ├── test
│   ├── test.json
│   ├── train.json
│   └── val.json
└── vqav2
    ├── answers
    ├── answers_upload
    ├── llava_vqav2_mscoco_test2015.jsonl
    ├── llava_vqav2_mscoco_test-dev2015.jsonl
    └── test2015
```


## Validating
Our image validation code comes from LLaVA, thanks for their contribution! 

You can refer to the official repository for validation, but we also provide [off-the-shelf](scripts/v1/eval) scripts.


### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `eval/vqav2`.
2. Multi-GPU inference.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/llava/vqav2.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/vqav2.sh
```

3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `eval/vqav2/answers_upload`.

### GQA

1. Download the data following the official instructions [here](https://cs.stanford.edu/people/dorarad/gqa/download.html) and put under `eval/gqa/data`.
2. Multi-GPU inference

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/llava/gqa.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/gqa.sh
```

### VisWiz

1. Download [`test.json`](https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip) and extract [`test.zip`](https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip) to `test`. Put them under `eval/vizwiz`.
2. Single-GPU inference.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/moe_llava/vizwiz.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/vizwiz.sh
```

3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/1911/my-submission): `eval/vizwiz/answers_upload`.

### ScienceQA

1. Under `eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/moe_llava/sqa.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/sqa.sh
```


### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `eval/textvqa`.
2. Single-GPU inference and evaluate.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/moe_llava/textvqa.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/textvqa.sh
```


### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `eval/pope`.
2. Single-GPU inference and evaluate.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/moe_llava/pope.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/pope.sh
```

### MME
1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. Put the official `eval_tool` and `MME_Benchmark_release_version` under `eval/MME`.
4. Single-GPU inference and evaluate.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/llava/mme.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/mme.sh
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `eval/mmbench`.
2. Single-GPU inference.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/llava/mmbench.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/mmbench.sh
```

3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `eval/mmbench/answers_upload/mmbench_dev_20230712`.


### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `eval/mmbench`.
2. Single-GPU inference.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/llava/mmbench_cn.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/mmbench_cn.sh
```

3. Submit the results to the [evaluation server](https://opencompass.org.cn/leaderboard-multimodal): `eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


### SEED-Bench

1. Following the official [instructions](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md) to download the images and the videos. Put images under `eval/seed_bench/SEED-Bench-image`.
2. Extract the video frame in the middle from the downloaded videos, and put them under `eval/seed_bench/SEED-Bench-video-image`.
3. Multiple-GPU inference and evaluate.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1/eval/llava/seed.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/seed.sh
```

4. Optionally, submit the results to the leaderboard: `eval/seed_bench/answers_upload` using the official jupyter notebook.



### LLaVA-Bench-in-the-Wild

1. Extract contents of [`llava-bench-in-the-wild`](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) to `eval/llava-bench-in-the-wild`.
2. Single-GPU inference and evaluate.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/moe_llava/llavabench.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/llavabench.sh
```


### MM-Vet

1. Extract [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `eval/mmvet`.
2. Single-GPU inference.

**LLaVA-based** model
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1/eval/moe_llava/mmvet.sh
```
**MoE-based** model
```Shell
bash scripts/v1/eval/moe_llava/mmvet.sh
```

