#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-One-Group-Out YOLO evaluator

功能：
- 针对 leave-one-out 版本的 app-ood / scene-ood 训练结果，
  自动遍历对应 split 目录和训练权重，调用 eval_yolo_center_hit_debug.py，
  解析每个 group 的 Success Rate，最后输出整体统计（mean / std）。

假设目录结构：
  splits_root/
    app-ood_1/
      test.txt
      labels_root.txt
    app-ood_2/
      test.txt
      labels_root.txt
    ...
  train_project/
    app-ood_idx0_1/
      weights/best.pt
    app-ood_idx1_2/
      weights/best.pt
    ...

使用示例（app-ood）：
  python leaveoneout_eval.py \
    --ood-type app-ood \
    --splits-dir splits_leave1out_app \
    --train-project runs/app_ood_leave1out \
    --eval-script eval_yolo_center_hit_debug.py \
    --class-id 0 --imgsz 640 --conf 0.05 --iou 0.6 --topk 1

使用示例（scene-ood）：
  python leaveoneout_eval.py \
    --ood-type scene-ood \
    --splits-dir splits_leave1out_scene \
    --train-project runs/scene_ood_leave1out \
    --eval-script eval_yolo_center_hit_debug.py \
    --class-id 0 --imgsz 640 --conf 0.05 --iou 0.6 --topk 1
"""

import argparse
import re
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_split_dirs_for_ood(splits_root: Path, ood_type: str) -> List[Path]:
    """
    在 splits_root 下查找当前 ood_type 对应的所有 split 目录。
    目录名形如：app-ood_11 / scene-ood_LivingRoom 等。
    """
    prefix = f"{ood_type}_"
    dirs = [
        d for d in splits_root.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    dirs.sort(key=lambda p: p.name)
    return dirs


def find_train_dir_for_group(
    train_project: Path,
    ood_type: str,
    safe_group: str,
) -> Optional[Path]:
    """
    在 train_project 下查找给定 group 对应的训练目录。

    假定训练目录名形如：
      app-ood_idx2_11
    即：以 "{ood_type}_idx" 开头，以 "_{safe_group}" 结尾。
    """
    prefix = f"{ood_type}_idx"
    suffix = f"_{safe_group}"

    candidates: List[Path] = []
    for d in train_project.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(prefix) and name.endswith(suffix):
            candidates.append(d)

    if not candidates:
        return None

    # 如有多个（比如重复训练），按名称排序选最后一个
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def run_single_eval(
    eval_script: Path,
    image_list: Path,
    labels_root: Path,
    weights: Path,
    class_id: int,
    imgsz: int,
    conf: float,
    iou: float,
    topk: int,
    project: Path,
    name: str,
    save_vis: bool = False,
) -> float:
    """
    调用 eval_yolo_center_hit_debug.py 做一次评测，并解析 Success Rate。
    """

    cmd = [
        "python",
        str(eval_script),
        "--image_list",
        str(image_list),
        "--labels-root",
        str(labels_root),
        "--weights",
        str(weights),
        "--class-id",
        str(class_id),
        "--imgsz",
        str(imgsz),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--topk",
        str(topk),
        "--project",
        str(project),
        "--name",
        name,
    ]
    if save_vis:
        cmd.append("--save-vis")

    print(f"[CMD] {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        print("[ERROR] eval 脚本返回非零退出码")
        print("-------- STDOUT --------")
        print(proc.stdout)
        print("-------- STDERR --------")
        print(proc.stderr)
        raise RuntimeError("eval 运行失败，请检查上方输出。")

    # 解析输出中的 `Success Rate           : XX.XX%`
    m = re.search(r"Success Rate\s*:\s*([0-9.]+)%", proc.stdout)
    if not m:
        print("-------- STDOUT --------")
        print(proc.stdout)
        raise RuntimeError("未在输出中找到 'Success Rate : xx.xx%' 行，无法解析准确率。")

    acc = float(m.group(1))
    print(f"[RESULT] Success Rate = {acc:.2f}%")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-One-Group-Out YOLO evaluator"
    )
    parser.add_argument(
        "--ood-type",
        type=str,
        required=True,
        choices=["app-ood", "scene-ood"],
        help="当前要评估的 OOD 类型。",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        required=True,
        help="split 根目录（里面包含 app-ood_xxx / scene-ood_xxx 等子目录）。",
    )
    parser.add_argument(
        "--train-project",
        type=str,
        required=True,
        help="训练输出的 project 目录（与训练脚本中的 --project 一致）。",
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default="eval_yolo_center_hit_debug.py",
        help="eval 脚本路径（默认当前目录下 eval_yolo_center_hit_debug.py）。",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="评估目标类别（默认 0）。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图像尺寸（默认 640）。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="预测的置信度阈值（默认 0.05）。",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS 的 IoU 阈值（默认 0.6）。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="中心点命中评测的 Top-K（默认 1，对应原逻辑）。",
    )
    parser.add_argument(
        "--eval-project",
        type=str,
        default="runs/eval-ctr-l1o",
        help="eval 输出的 project 目录（默认 runs/eval-ctr-l1o）。",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="是否保存可视化结果（调试时可开启，比较占空间）。",
    )

    args = parser.parse_args()

    splits_root = Path(args.splits_dir)
    train_project = Path(args.train_project)
    eval_script = Path(args.eval_script)
    eval_project = Path(args.eval_project)

    if not splits_root.exists():
        raise FileNotFoundError(f"splits-dir 不存在：{splits_root}")
    if not train_project.exists():
        raise FileNotFoundError(f"train-project 不存在：{train_project}")
    if not eval_script.exists():
        raise FileNotFoundError(f"eval 脚本不存在：{eval_script}")

    ood_type = args.ood_type

    # 1. 找到当前 ood_type 对应的所有 split 目录
    split_dirs = find_split_dirs_for_ood(splits_root, ood_type)
    if not split_dirs:
        raise RuntimeError(
            f"在 {splits_root} 下找不到以 {ood_type + '_'} 开头的 split 目录，"
            f"请确认 --splits-dir 和 --ood-type 是否正确。"
        )

    print(f"[INFO] 在 {splits_root} 下找到 {len(split_dirs)} 个 split 目录用于评估：")
    for d in split_dirs:
        print(f"  - {d.name}")

    accuracies: List[float] = []
    group_names: List[str] = []
    missing_weights: List[str] = []

    # 2. 逐个 split 进行评估
    for split_dir in split_dirs:
        name = split_dir.name
        # 名字形如 "app-ood_11"，safe_group = "11"
        safe_group = name[len(ood_type) + 1 :]
        group_names.append(safe_group)

        print("\n" + "=" * 60)
        print(f"[INFO] 评估 split 目录: {split_dir} (group={safe_group})")

        test_list = split_dir / "test.txt"
        if not test_list.exists():
            print(f"[WARN] 跳过：缺少 test.txt → {test_list}")
            continue

        # labels_root 优先从 labels_root.txt 中读取
        labels_root_txt = split_dir / "labels_root.txt"
        if labels_root_txt.exists():
            labels_root = Path(labels_root_txt.read_text(encoding="utf-8").strip())
            print(f"[INFO] 使用 labels_root (from labels_root.txt): {labels_root}")
        else:
            print(f"[WARN] {labels_root_txt} 不存在，无法确定 labels_root，跳过该 split。")
            continue

        # 在 train_project 下找到对应的训练目录
        train_dir = find_train_dir_for_group(train_project, ood_type, safe_group)
        if train_dir is None:
            print(
                f"[WARN] 未找到对应训练目录（前缀 {ood_type}_idx、后缀 _{safe_group}），"
                f"该 group 将不计入平均值。"
            )
            missing_weights.append(safe_group)
            continue

        weights = train_dir / "weights" / "best.pt"
        if not weights.exists():
            print(f"[WARN] 训练目录存在，但缺少 best.pt：{weights}，该 group 将不计入平均值。")
            missing_weights.append(safe_group)
            continue

        print(f"[INFO] 使用模型权重: {weights}")

        # eval 的 run name 可以用 group 标识
        eval_name = f"{ood_type}_{safe_group}"

        acc = run_single_eval(
            eval_script=eval_script,
            image_list=test_list,
            labels_root=labels_root,
            weights=weights,
            class_id=args.class_id,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            topk=args.topk,
            project=eval_project,
            name=eval_name,
            save_vis=args.save_vis,
        )
        accuracies.append(acc)

    # 3. 汇总结果
    print("\n" + "#" * 60)
    print(f"[SUMMARY] OOD type = {ood_type}")
    print(f"  总 split 数量      : {len(split_dirs)}")
    print(f"  实际评估成功的数量 : {len(accuracies)}")

    for g, acc in zip(group_names, accuracies):
        print(f"  Group={g:>6s} : {acc:.2f}%")

    if missing_weights:
        print("\n[WARN] 以下 group 没有找到对应的模型或 best.pt，未计入平均值：")
        print("       " + ", ".join(missing_weights))

    if accuracies:
        mean_acc = statistics.mean(accuracies)
        if len(accuracies) > 1:
            std_acc = statistics.pstdev(accuracies)
            print(f"\n  Mean accuracy : {mean_acc:.2f}%")
            print(f"  Std  accuracy : {std_acc:.2f}%")
        else:
            print(f"\n  Mean accuracy : {mean_acc:.2f}%（仅 1 个 group，不计算 std）")
    else:
        print("\n[ERROR] 没有任何 group 被成功评估，请检查上面的 WARN / ERROR 信息。")


if __name__ == "__main__":
    main()
