#!/home/stud/yyang/jiameng/env/bin/bash

#SBATCH --job-name fed
#SBATCH --output=/home/stud/yyang/jiameng/slurm_out/test.out
#SBATCH --ntasks=1
#SBATCH --time=200:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --error=/home/stud/yyang/jiameng/slurm_out/test.e
#SBATCG --mail-type=ALLs

source /home/stud/yyang/jiameng/env/bin/activate
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/okvqa_ft_qformer_t5.yaml
