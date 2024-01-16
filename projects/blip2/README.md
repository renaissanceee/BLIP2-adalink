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
### git
current commit in meng branch
```python
git clone -b meng https://github.com/YaoZ720/VLM_Fed.git
```
### env
recommend: torch 2.0.0+cu118 
```python
python -m venv env %create
.\env\Scripts\activate %activate
(linux)source .\env\Scripts\activate %activate
pip install -r requirements.txt
```
### automatic download
```python
cp ./lavis/datasets/download_scripts/download_coco.py .
python download_coco.py
```

### run
```python
python train.py --cfg-path lavis/projects/blip2/train/okvqa_ft.yaml
# for distributed train
sbatch run_scripts/blip2/train/train_okvqa.sh

```
### adjust param
"./lavis/projects/blip2/train/okvqa_ft.yaml"

```
model: freeze_qformer, freeze_linear, ada_rank
run: resume_ckpt_path, output_dir,batch_size_train
```
"./lavis/models/blip2_t5.py"

1.how to set linear or adalink:```line 100```

2.how to set wandb:```line 198```

3.how to change input_tokens for vqa/caption:```line 143,151```
