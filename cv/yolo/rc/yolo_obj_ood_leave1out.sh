#!/bin/bash -l

#SBATCH --job-name=Yolo_object_ood_10fold
#SBATCH --mail-user=xy3371@g.rit.edu
#SBATCH --mail-type=ALL
#SBATCH --error=/home/xy3371/Yolo/RC_error/err_%A_%a.txt
#SBATCH --output=/home/xy3371/Yolo/RC_out/out_%A_%a.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=60:00:00
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --account=playback
#SBATCH --array=0-9

conda activate yolo

IDX=${SLURM_ARRAY_TASK_ID}

python yolo_train_leave_one_group.py \
  --data-csv data_stat.csv \
  --ood-type object-ood \
  --group-col Object \
  --object-folds 10 \
  --kfold-index ${IDX} \
  --images-root datasets/myar/images \
  --labels-root datasets/myar/labels \
  --model /home/xy3371/Yolo/weights/yolo12n.pt \
  --epochs 150 \
  --imgsz 640 \
  --project runs/object_ood_10fold \
  --splits-dir .split_leave1out_object \
  --seed 0
