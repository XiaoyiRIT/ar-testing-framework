#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成工程结构的 Markdown 文件（项目“快照说明”），便于在对话中共享上下文。
- 输出：默认写入 project_context.md
- 目录树：可设最大深度
- 排除：默认忽略 .git、虚拟环境、缓存与大文件等，可自定义
- 可选摘录：按通配符挑选关键文件，截取前 N 行作为上下文片段

python gen_project_context.py --out chat_context.md --tree-depth 4

"""

import argparse
import datetime as dt
import os
import sys
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

# 目录名级别的硬排除（出现即跳过整个子树）
EXCLUDE_DIR_NAMES = {".git", ".svn", ".hg", ".venv", ".venv311", "venv", "env", "ENV", ".conda", "conda"}
# 文件后缀级别的硬排除（可选）
EXCLUDE_FILE_SUFFIXES = {".pt", ".onnx", ".ckpt", ".mp4", ".mov", ".avi", ".zip", ".tar", ".gz", ".7z"}

DEFAULT_EXCLUDES = [
    ".git", ".svn", ".hg",
    ".DS_Store",
    "__pycache__", "*.pyc", "*.pyo", "*.pyd",
    ".idea", ".vscode",
    ".venv", ".venv311", "venv", "env", "ENV", ".conda", "conda", ".env",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules",
    "runs", "outputs", "logs",
    # 大文件常见后缀（仅用于目录树与统计时忽略；不影响真实代码）
    "*.pt", "*.onnx", "*.ckpt",
    "*.mp4", "*.mov", "*.avi",
    "*.png", "*.jpg", "*.jpeg", "*.webp",
    "*.zip", "*.tar", "*.gz", "*.7z",
]

DEFAULT_SNIPPETS = [
    "README.md",
    "*.md",
    "*.py",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
]

def parse_args():
    p = argparse.ArgumentParser(description="生成工程结构 Markdown 文档")
    p.add_argument("--root", type=str, default=".", help="工程根目录（默认当前目录）")
    p.add_argument("--out", type=str, default="project_context.md", help="输出文件名")
    p.add_argument("--tree-depth", type=int, default=3, help="目录树最大深度（默认 3）")
    p.add_argument("--exclude", action="append", default=[], help="追加排除模式（可多次）")
    p.add_argument("--no-default-excludes", action="store_true", help="不使用默认排除列表")
    p.add_argument("--snippet-glob", action="append", default=[], help="需要摘录片段的通配符（可多次）")
    p.add_argument("--snippet-lines", type=int, default=120, help="每个文件摘录的最大行数")
    p.add_argument("--topn", type=int, default=15, help="体积 Top-N 文件列表条数")
    return p.parse_args()

def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()

def run_cmd(cmd: List[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True, timeout=5)
        return out.strip()
    except Exception:
        return ""

def gather_git_info(root: Path) -> Tuple[str, str, str]:
    if not is_git_repo(root):
        return ("N/A", "N/A", "N/A")
    branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], root) or "N/A"
    commit = run_cmd(["git", "rev-parse", "--short", "HEAD"], root) or "N/A"
    latest = run_cmd(["git", "log", "--oneline", "-n", "5"], root)
    return (branch, commit, latest or "N/A")

def match_any(path: Path, patterns: Iterable[str]) -> bool:
    from fnmatch import fnmatch
    rel = str(path).replace("\\", "/")
    name = path.name
    # 1) 通配匹配：支持 **/dir/** 形式
    if any(fnmatch(rel, pat) or fnmatch(name, pat) for pat in patterns):
        return True
    # 2) 目录名硬排除：只要任何一段路径命中就排除
    for part in path.parts:
        if part in EXCLUDE_DIR_NAMES:
            return True
    # 3) 文件后缀硬排除
    if path.is_file() and path.suffix.lower() in EXCLUDE_FILE_SUFFIXES:
        return True
    return False

def iter_tree(root: Path, max_depth: int, excludes: List[str]) -> List[str]:
    lines = []
    root = root.resolve()

    def walk(d: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        # 先分组，目录要能被“剪枝”
        dirs, files = [], []
        for e in entries:
            if e.is_dir():
                # 如果 e 这个目录命中排除，直接跳过（剪枝）
                if match_any(e.relative_to(root), excludes) or e.name in EXCLUDE_DIR_NAMES:
                    continue
                dirs.append(e)
            else:
                if match_any(e.relative_to(root), excludes):
                    continue
                files.append(e)

        # 打印目录
        for i, e in enumerate(dirs + files):
            is_last = (i == len(dirs + files) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{e.name}{'/' if e.is_dir() else ''}")
            if e.is_dir():
                new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                walk(e, depth + 1, new_prefix)

    lines.append(f"{root.name}/")
    walk(root, 1, "")
    return lines


def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units)-1:
        f /= 1024.0
        i += 1
    return f"{f:.1f} {units[i]}"

def list_files(root: Path, excludes: List[str]) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        # 先对目录剪枝：只要路径包含被排除目录名，整条路径跳过
        if any(part in EXCLUDE_DIR_NAMES for part in p.parts):
            continue
        if p.is_file() and not match_any(p.relative_to(root), excludes):
            files.append(p)
    return files


def pick_snippet_files(root: Path, patterns: List[str], excludes: List[str]) -> List[Path]:
    from fnmatch import fnmatch
    files = []
    if not patterns:
        return files
    for p in root.rglob("*"):
        if p.is_file() and not match_any(p.relative_to(root), excludes):
            if any(fnmatch(p.name, pat) or fnmatch(str(p.relative_to(root)), pat) for pat in patterns):
                files.append(p)
    # 去重 & 稳定排序
    seen = set()
    res = []
    for p in sorted(files, key=lambda x: str(x).lower()):
        k = str(p)
        if k not in seen:
            res.append(p)
            seen.add(k)
    return res

def main():
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.exists():
        print(f"[ERR] 路径不存在：{root}", file=sys.stderr)
        sys.exit(1)

    excludes = []
    if not args.no_default_excludes:
        excludes.extend(DEFAULT_EXCLUDES)
    excludes.extend(args.exclude or [])

    # 基本信息
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    branch, commit, latest = gather_git_info(root)

    # 目录树
    tree_lines = iter_tree(root, args.tree_depth, excludes)

    # 文件统计
    files = list_files(root, excludes)
    total_size = sum(p.stat().st_size for p in files) if files else 0
    topn = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[: max(args.topn, 0)]

    # 摘录文件
    snippet_globs = args.snippet_glob or DEFAULT_SNIPPETS
    snippet_files = pick_snippet_files(root, snippet_globs, excludes)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# Project Context\n\n")
        f.write(f"- **Path**: `{root}`\n")
        f.write(f"- **Time (UTC)**: {now}\n")
        f.write(f"- **Git Branch**: {branch}\n")
        f.write(f"- **Git Commit**: {commit}\n\n")

        f.write(f"## Directory Tree (depth={args.tree_depth})\n\n")
        f.write("```\n")
        for line in tree_lines:
            f.write(line + "\n")
        f.write("```\n\n")

        f.write("## File Stats\n\n")
        f.write(f"- Files counted (excluded patterns applied): **{len(files)}**\n")
        f.write(f"- Total size: **{human_bytes(total_size)}**\n\n")

        if topn:
            f.write(f"### Top {len(topn)} Largest Files (excluded patterns applied)\n\n")
            f.write("| Size | Path |\n|---:|---|\n")
            for p in topn:
                size = human_bytes(p.stat().st_size)
                rel = p.relative_to(root)
                f.write(f"| {size} | `{rel}` |\n")
            f.write("\n")

        if is_git_repo(root):
            f.write("## Recent Commits\n\n")
            f.write("```\n")
            f.write(latest + ("\n" if latest else ""))
            f.write("```\n\n")

        if snippet_files:
            f.write(f"## File Snippets (first {args.snippet_lines} lines)\n\n")
            for p in snippet_files:
                rel = p.relative_to(root)
                f.write(f"### `{rel}`\n\n")
                f.write("```text\n")
                try:
                    with p.open("r", encoding="utf-8", errors="replace") as fp:
                        for i, line in enumerate(fp):
                            if i >= args.snippet_lines:
                                f.write("... (truncated)\n")
                                break
                            f.write(line.rstrip("\n") + "\n")
                except Exception as e:
                    f.write(f"[Cannot read file: {e}]\n")
                f.write("```\n\n")

        f.write("## Excludes\n\n")
        f.write("```\n")
        for e in excludes:
            f.write(str(e) + "\n")
        f.write("```\n")

    print(f"[OK] Wrote {out} (root={root})")

if __name__ == "__main__":
    main()
