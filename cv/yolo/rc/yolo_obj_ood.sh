#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=Yolo_random
#SBATCH --mail-user=xy3371@g.rit.edu
#SBATCH --mail-type=ALL
#SBATCH --error=/home/xy3371/Yolo/RC_error/err_%j.txt
#SBATCH --output=/home/xy3371/Yolo/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --account=playback

conda activate yolo

GLOG_vmodule=MemcachedClient=-1 \

#Train
srun python test_yolo_train_dynsplit.py train \
  --images-root datasets/myar/images \
  --labels-root datasets/myar/labels \
  --data-csv data_stat.csv \
  --split-mode object-ood --group-col Object \
  --ratios 0.7,0.15,0.15 \
  --weights /home/xy3371/Yolo/weights/yolo11n.pt --epochs 150 --imgsz 640 \
  --split-outdir .split
