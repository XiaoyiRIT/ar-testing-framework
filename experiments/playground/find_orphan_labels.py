# -*- coding: utf-8 -*-
"""
在指定 labels_dir 下递归查找“没有对应同名图片”的 .txt 标签文件。
- 默认只列出这些“孤儿 label”
- 传 --delete 则会直接删除它们
- 支持常见图片扩展名：.jpg .jpeg .png .bmp .webp

用法：
  仅列出：
    python find_orphan_labels.py myar/images/train myar/labels/train
  直接删除：
    python find_orphan_labels.py /path/to/labels /path/to/images --delete
"""

import sys
from pathlib import Path
from typing import Set, Tuple

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_image_stems(images_root: Path) -> Set[str]:
    stems: Set[str] = set()
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stems.add(p.stem.lower())
    return stems

def find_orphan_labels(labels_root: Path, images_root: Path, do_delete: bool = False) -> Tuple[int, int]:
    image_stems = collect_image_stems(images_root)
    if not image_stems:
        print(f"WARNING: images_dir `{images_root}` 中未发现任何图片文件。")

    total, orphans = 0, 0
    for txt in labels_root.rglob("*.txt"):
        total += 1
        if txt.stem.lower() not in image_stems:
            orphans += 1
            if do_delete:
                try:
                    txt.unlink()
                    print(f"DELETE: {txt}")
                except Exception as e:
                    print(f"ERROR deleting {txt}: {e}")
            else:
                print(f"ORPHAN: {txt}")

    return orphans, total

def main() -> None:
    if len(sys.argv) < 3:
        print("用法：python find_orphan_labels.py <labels_dir> <images_dir> [--delete]")
        sys.exit(1)

    labels_dir = Path(sys.argv[1]).resolve()
    images_dir = Path(sys.argv[2]).resolve()
    do_delete = (len(sys.argv) >= 4 and sys.argv[3] == "--delete")

    if not labels_dir.is_dir():
        print(f"❌ labels_dir 无效：{labels_dir}")
        sys.exit(1)
    if not images_dir.is_dir():
        print(f"❌ images_dir 无效：{images_dir}")
        sys.exit(1)

    orphans, total = find_orphan_labels(labels_dir, images_dir, do_delete)
    action = "删除" if do_delete else "发现"
    print(f"\n✅ 完成：{action} 孤儿 label {orphans} 个 / 标签总数 {total} 个。")

if __name__ == "__main__":
    main()
