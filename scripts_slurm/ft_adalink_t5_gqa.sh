#!/usr/bin/env bash

#SBATCH --job-name fed
#SBATCH --output=/nfs/data2/zhangya/slurm/test_ada_t5_gqa.out
#SBATCH --ntasks=1
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --error=/nfs/data2/zhangya/slurm/test_ada_t5_gqa.e
#SBATCH --nodelist=worker-6

source /home/stud/zhangya/jiameng/BLIP2-adalink/env/bin/activate
python -m torch.distributed.run --nproc_per_node=1 /home/stud/zhangya/jiameng/BLIP2-adalink/train.py --cfg-path /home/stud/zhangya/jiameng/BLIP2-adalink/config_blip2/train/gqa_ft_t5.yaml