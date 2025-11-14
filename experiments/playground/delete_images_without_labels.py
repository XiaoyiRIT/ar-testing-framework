# -*- coding: utf-8 -*-
"""
删除没有对应 label 的图片：
- 支持递归遍历 images_dir
- 支持 labels_dir 与 images_dir 分离（YOLO 常见结构）
- 匹配规则：图片 basename == 标签 basename（忽略扩展名与大小写）
- 识别的图片扩展名：.jpg .jpeg .png .bmp .webp

用法示例：
python delete_images_without_labels.py myar/images/train myar/labels/train
# 若 labels 就在同一目录（旁边同名 .txt），则：
python delete_images_without_labels.py /path/to/images
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Set

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_label_stems(labels_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in labels_root.rglob("*.txt"):
        stems.add(p.stem.lower())
    return stems

def delete_orphan_images(images_root: Path, labels_root: Optional[Path] = None) -> Tuple[int, int]:
    deleted, kept = 0, 0

    if labels_root is None:
        # 同目录：图片旁边找同名 .txt
        for img in images_root.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                txt = img.with_suffix(".txt")
                if not txt.exists():
                    try:
                        img.unlink()
                        deleted += 1
                        print(f"DELETE: {img}")
                    except Exception as e:
                        print(f"ERROR deleting {img}: {e}")
                else:
                    kept += 1
        return deleted, kept

    # 分离目录：预收集 labels 的 basename
    label_stems = collect_label_stems(labels_root)
    if not label_stems:
        print(f"WARNING: labels_dir `{labels_root}` 中未发现任何 .txt 标签文件。")

    for img in images_root.rglob("*"):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
            if img.stem.lower() not in label_stems:
                try:
                    img.unlink()
                    deleted += 1
                    print(f"DELETE: {img}")
                except Exception as e:
                    print(f"ERROR deleting {img}: {e}")
            else:
                kept += 1

    return deleted, kept

def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python delete_images_without_labels.py <images_dir> [labels_dir]")
        sys.exit(1)

    images_dir = Path(sys.argv[1]).resolve()
    labels_dir = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else None

    if not images_dir.is_dir():
        print(f"❌ images_dir 无效：{images_dir}")
        sys.exit(1)
    if labels_dir is not None and not labels_dir.is_dir():
        print(f"❌ labels_dir 无效：{labels_dir}")
        sys.exit(1)

    deleted, kept = delete_orphan_images(images_dir, labels_dir)
    print(f"\n✅ 处理完成：删除 {deleted} 张图片，保留 {kept} 张图片。")

if __name__ == "__main__":
    main()
