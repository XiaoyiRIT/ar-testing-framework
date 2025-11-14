# -*- coding: utf-8 -*-
"""
诊断 labels 比 images 多 1 的原因：
- 递归收集两边的“基名集合”（忽略扩展名、大小写）
- 打印数量统计与差集
- 兼容 .TXT/.JPG 等大小写
- 可选择忽略诸如 classes.txt 等“非标注”文本

用法：
  仅诊断：
    python diagnose_orphan_labels.py /path/to/labels /path/to/images
  忽略某些文件名（逗号分隔，基名或完整文件名都可）：
    python diagnose_orphan_labels.py myar/images/train myar/labels/train --ignore classes.txt,readme.txt
"""

import sys
from pathlib import Path
from typing import Set, List

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT: str = ".txt"

def parse_ignore_list(arg: str) -> Set[str]:
    items = set()
    for s in arg.split(","):
        s = s.strip()
        if s:
            items.add(s.lower())
    return items

def collect_image_stems(images_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stems.add(p.stem.lower())
    return stems

def collect_label_stems(labels_root: Path, ignore: Set[str]) -> Set[str]:
    stems: Set[str] = set()
    for p in labels_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == LABEL_EXT:
            # 忽略名单支持：完整文件名 或 仅基名
            name_l = p.name.lower()
            stem_l = p.stem.lower()
            if name_l in ignore or stem_l in ignore:
                continue
            stems.add(stem_l)
    return stems

def main() -> None:
    if len(sys.argv) < 3:
        print("用法：python diagnose_orphan_labels.py <labels_dir> <images_dir> [--ignore name1,name2]")
        sys.exit(1)

    labels_dir = Path(sys.argv[1]).resolve()
    images_dir = Path(sys.argv[2]).resolve()
    ignore: Set[str] = set()
    if len(sys.argv) >= 4 and sys.argv[3].startswith("--ignore"):
        parts = sys.argv[3].split("=", 1)
        if len(parts) == 2:
            ignore = parse_ignore_list(parts[1])
        elif len(sys.argv) >= 5:
            ignore = parse_ignore_list(sys.argv[4])

    if not labels_dir.is_dir():
        print(f"❌ labels_dir 无效：{labels_dir}")
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"❌ images_dir 无效：{images_dir}")
        sys.exit(1)

    img_stems = collect_image_stems(images_dir)
    lab_stems = collect_label_stems(labels_dir, ignore)

    print("—— 统计 ——")
    print(f"images_dir: {images_dir}")
    print(f"labels_dir: {labels_dir}")
    print(f"忽略名单: {sorted(ignore) if ignore else '（无）'}")
    print(f"图片数量（按基名去重）: {len(img_stems)}")
    print(f"标签数量（按基名去重）: {len(lab_stems)}")
    diff_lab = sorted(lab_stems - img_stems)
    diff_img = sorted(img_stems - lab_stems)

    print("\n—— labels有而images没有（疑似多余label）——")
    if diff_lab:
        for s in diff_lab[:50]:
            print("  +", s)
        if len(diff_lab) > 50:
            print(f"  ... 以及 {len(diff_lab)-50} 个更多")
    else:
        print("  无")

    print("\n—— images有而labels没有（疑似缺失label的图片）——")
    if diff_img:
        for s in diff_img[:50]:
            print("  -", s)
        if len(diff_img) > 50:
            print(f"  ... 以及 {len(diff_img)-50} 个更多")
    else:
        print("  无")

    # 小结判断
    print("\n—— 结论 ——")
    if diff_lab:
        print(f"labels 确实多出 {len(diff_lab)} 个基名。")
    elif diff_img:
        print(f"labels 并不比 images 多；相反缺少 {len(diff_img)} 个标签。")
    else:
        print("两边基名数量一致。如果你仍看到“文件数相差 1”，很可能是被计入了非图片/非标签文件（如 classes.txt、README.txt、.DS_Store），或统计时跨了 train/val 分区。")

if __name__ == "__main__":
    main()
