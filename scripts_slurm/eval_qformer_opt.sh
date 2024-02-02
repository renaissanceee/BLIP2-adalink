#!/usr/bin/env bash

#SBATCH --job-name fed
#SBATCH --output=eval_qforemr_opt.out
#SBATCH --ntasks=1
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --error=eval_qforemr_opt.e
#SBATCH --nodelist=worker-7

source ./env/bin/activate
python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path config_blip2/eval/okvqa_ft_qformer_opt.yaml