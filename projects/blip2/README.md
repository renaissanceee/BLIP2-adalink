## BLIP-2


### Image-Text Matching Example
BLIP-2 can compute the image-text matching score using the same interface as BLIP. Checkout this [notebook](https://github.com/salesforce/LAVIS/blob/3446bac20c5646d35ae383ebe6d13cec4f8b00cb/examples/blip2_image_text_matching.ipynb) for an example. 

### Benchmark Evaluation 
Follow [Dataset Download](https://opensource.salesforce.com/LAVIS//latest/getting_started.html#auto-downloading-and-loading-datasets) to prepare common vision-language datasets.

Run [these scripts](https://github.com/salesforce/LAVIS/tree/main/run_scripts/blip2/eval) for evaluating pretrained and finetuned models. 

### Training
Stage-1 Pre-training (from scratch): 
```bash run_scripts/blip2/train/pretrain_stage1.sh```

Stage-2 Pre-training: 
```bash run_scripts/blip2/train/pretrain_stage2.sh```

Finetune for image captioning: 
```bash run_scripts/blip2/train/train_caption_coco.sh```

The [config files](https://github.com/salesforce/LAVIS/tree/main/lavis/projects/blip2/train) can be modified for customized training.

## hints for adalink version

### env
recommand: torch 2.0.0+cu118 
```python
python-m venv env %create
.\env\Scripts\activate %activate
(linux)source .\env\Scripts\activate %activate
pip install -r requirements.txt
```
### run
```python
python train.py --cfg-path lavis/projects/blip2/train/okvqa_ft.yaml
```
### adjust param
```./lavis/projects/blip2/train/okvqa_ft.yaml```

1.how to set resume:```run.resume_ckpt_path=null```

2.where to save ckpt:```run.output_dir="output/BLIP2/OKVQA"```

3.how to set batch_size:```run.batch_size_train=16```

```./lavis/models/blip2_t5.py```

1.freeze_vit,freeze_qformer:```line 48,49```

2.how to set rank of adalink:```line 99```

3.how to set linear or adalink:```line 100```

4.how to set wandb:```line 198```

5.how to change input_tokens for vqa/caption:```line 143,151```
