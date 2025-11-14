#!/usr/bin/env python3
import argparse, subprocess, time, re, sys

PKG = "com.google.ar.core.examples.java.hellorecordingplayback"
ACT = ".HelloRecordingPlaybackActivity"  # 也可用完整名：com.google.ar.core.examples.java.hellorecordingplayback.HelloRecordingPlaybackActivity

def run(cmd):
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8","ignore")

def start_app(serial=None):
    base = f"adb -s {serial} " if serial else "adb "
    # 启动：包名/Activity 支持以点开头的简写
    run(base + f"shell am start -n {PKG}/{ACT}")

def top_component(serial=None):
    base = f"adb -s {serial} " if serial else "adb "
    # 优先从 activity dumpsys 抓 topResumedActivity，抓不到再看 window 的 mCurrentFocus
    out = run(base + "shell dumpsys activity")
    m = re.search(r'topResumedActivity.*? ([\w\.]+)/([\w\.$]+)', out)
    if not m:
        m = re.search(r'mResumedActivity.*? ([\w\.]+)/([\w\.$]+)', out)
    if m:
        return m.group(1), m.group(2)
    out = run(base + "shell dumpsys window")
    m = re.search(r'mCurrentFocus.*? ([\w\.]+)/([\w\.$]+)', out)
    if m:
        return m.group(1), m.group(2)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", help="设备序列号（adb devices 可查看，多设备时必须指定）")
    args = ap.parse_args()

    # 启动 App
    start_app(args.serial)
    time.sleep(3.0)  # 等摄像头/AR会话初始化

    # 校验前台是否在目标包
    tpkg, tact = top_component(args.serial)
    if tpkg is None:
        print("✖ 读取前台 Activity 失败（权限或系统限制）。")
        sys.exit(2)

    print(f"前台：{tpkg}/{tact}")
    if tpkg == PKG:
        print("✓ SMOKE OK：App 已启动并处于前台。")
        sys.exit(0)
    else:
        print("✖ SMOKE FAIL：前台不在目标包。")
        sys.exit(1)

if __name__ == "__main__":
    main()

  