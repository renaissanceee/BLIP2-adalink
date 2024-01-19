#!/bin/bash
source /home/stud/yyang/jiameng/env/bin/activate
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/okvqa_ft_qformer_t5.yaml