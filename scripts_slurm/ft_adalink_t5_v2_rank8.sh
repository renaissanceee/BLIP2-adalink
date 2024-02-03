#!/usr/bin/env bash

#SBATCH --job-name fed
#SBATCH --output=test_ada_t5_v2_rank8.out
#SBATCH --ntasks=1
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --error=test_ada_t5_v2_rank8.e
#SBATCH --nodelist=worker-6

source /home/stud/zhangya/jiameng/BLIP2-adalink/env/bin/activate
python -m torch.distributed.run --nproc_per_node=1 /home/stud/zhangya/jiameng/BLIP2-adalink/train.py --master_port 12332 --cfg-path /home/stud/zhangya/jiameng/BLIP2-adalink/config_blip2/train/vqav2_ft_t5_rank8.yaml