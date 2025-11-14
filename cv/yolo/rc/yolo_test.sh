#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=Yolo_test
#SBATCH --mail-user=xy3371@g.rit.edu
#SBATCH --mail-type=ALL
#SBATCH --error=/home/xy3371/Yolo/RC_error/err_%j.txt
#SBATCH --output=/home/xy3371/Yolo/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --account=playback

conda activate yolo

GLOG_vmodule=MemcachedClient=-1 \

#Train
srun python Yolo/test_yolo_train.py train --data Yolo/myar_scene.yaml --weights yolo11s.pt --epochs 200 --imgsz 640
#Val
#srun python Yolo/test_yolo_train.py val --data myar2.yaml --weights runs/train/exp/weights/best.pt
#Test
#srun python Yolo/test_yolo_train.py test --test-dir testimages --weights runs/train/exp4/weights/best.pt --conf 0.05

