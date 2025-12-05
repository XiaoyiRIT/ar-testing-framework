#!/bin/bash -l

#SBATCH --job-name=Yolo_scene_ood_k5
#SBATCH --mail-user=xy3371@g.rit.edu
#SBATCH --mail-type=ALL
#SBATCH --error=/home/xy3371/Yolo/RC_error/err_%A_%a.txt
#SBATCH --output=/home/xy3371/Yolo/RC_out/out_%A_%a.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --account=playback
#SBATCH --array=0-4   # ⭐ 关键：5 个子任务，对应 fold 0~4

conda activate yolo

FOLD_ID=${SLURM_ARRAY_TASK_ID}

python test_yolo_train_dynsplit.py train \
  --images-root datasets/myar/images \
  --labels-root datasets/myar/labels \
  --data-csv data_stat.csv \
  --split-mode scene-ood --group-col Scene \
  --ratios 0.7,0.15,0.15 \
  --kfold 5 --fold-index ${FOLD_ID} \
  --weights /home/xy3371/Yolo/weights/yolo12n.pt --epochs 150 --imgsz 640 \
  --split-outdir .split
