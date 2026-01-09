#!/bin/bash -l

#SBATCH --job-name=Yolo_full_train
#SBATCH --mail-user=xy3371@g.rit.edu
#SBATCH --mail-type=ALL
#SBATCH --error=/home/xy3371/Yolo/RC_error/err_%j.txt
#SBATCH --output=/home/xy3371/Yolo/RC_out/out_%j.txt
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --account=playback

# 环境
conda activate yolo

# 进入你的工程目录（按需修改）
# cd /home/xy3371/Yolo/your_project_dir

# 全量训练（默认 val=False，不做 evaluation/validation）
python yolo_train_full.py \
  --data-csv data_stat.csv \
  --images-root datasets/myar/images \
  --labels-root datasets/myar/labels_2 \
  --model /home/xy3371/Yolo/weights/yolo12n.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --seed 0 \
  --project runs/train_full \
  --name full_yolo12n_s0 \
  --splits-dir .split_full

# 若你希望训练过程中也跑验证（可选），加上 --val
#   --val
