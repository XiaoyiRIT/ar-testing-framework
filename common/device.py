# common/device.py
# ------------------------------------------------------------
# Description:
#   Appium driver bootstrap (UiAutomator2Options) + adb helpers
#   + simple utilities (window size, screenshot if需要扩展).
#   Mirrors your working connection semantics.
# ------------------------------------------------------------

import subprocess, cv2, numpy as np, time
from typing import Optional
from appium import webdriver
from appium.options.android import UiAutomator2Options
from selenium.common.exceptions import WebDriverException


def _try_connect(url, options):
    try:
        drv = webdriver.Remote(url, options=options)
        return drv
    except WebDriverException as e:
        # 典型 404/UnknownCommandError：base-path 不对
        msg = str(e)
        if "UnknownCommandError" in msg or "requested resource could not be found" in msg or "404" in msg:
            return None
        # 其他异常直接抛
        raise

def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8", "ignore")

def adb(cmd: str, serial: Optional[str] = None) -> str:
    base = f"adb -s {serial} " if serial else "adb "
    return sh(base + cmd)

def one_device_or_none() -> Optional[str]:
    out = sh("adb devices")
    lines = [l for l in out.strip().splitlines() if "\tdevice" in l and not l.startswith("List")]
    return lines[0].split("\t")[0] if len(lines) == 1 else None

def resolve_main_activity(pkg: str, serial: Optional[str] = None) -> Optional[str]:
    intent = 'intent:#Intent;action=android.intent.action.MAIN;category=android.intent.category.LAUNCHER;end'
    try:
        out = adb(f'shell cmd package query-activities --brief "{intent}"', serial)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith(pkg + "/"):
                return line.split("/", 1)[1]
    except Exception:
        pass
    return None

def make_driver(pkg: str,
                activity: Optional[str],
                serial: Optional[str],
                warmup_wait: float = 3.0):
    """Create driver with your v0-style capabilities."""
    caps = {
        "platformName": "Android",
        "appium:automationName": "UiAutomator2",
        "appium:deviceName": serial or (one_device_or_none() or "Android"),
        "appium:udid": serial if serial else one_device_or_none(),
        "appium:noReset": True,
        "appium:autoGrantPermissions": True,
        "appium:newCommandTimeout": 300,
        "appium:appPackage": pkg,
        "appium:disableWindowAnimation": True,
    }

    if activity and activity not in ("auto", None):
        caps["appium:appActivity"] = activity
        caps["appium:appWaitActivity"] = "*"
    else:
        caps["appium:intentAction"]   = "android.intent.action.MAIN"
        caps["appium:intentCategory"] = "android.intent.category.LAUNCHER"
        caps["appium:intentFlags"]    = "0x10200000"
        caps["appium:appWaitActivity"] = "*"

    options = UiAutomator2Options().load_capabilities(caps)
    
    # 允许从环境或 config 里传 server_url；没有就用默认
    server_url = "http://127.0.0.1:4723"  # 你也可以从 config 传进来
    # 组装两条候选：根路径 与 /wd/hub（去重）
    candidates = []
    if server_url.rstrip("/").endswith("/wd/hub"):
        root = server_url.rstrip("/").rsplit("/wd/hub", 1)[0]
        candidates = [server_url, root]
    else:
        hub = server_url.rstrip("/") + "/wd/hub"
        candidates = [server_url, hub]

    last_err = None
    for url in candidates:
        drv = _try_connect(url, options)
        if drv is not None:
            print(f"[device] connected via {url}")
            time.sleep(warmup_wait)
            return drv
    # 两条都失败，抛出更友好的错误
    raise RuntimeError(
        f"Failed to create session. Tried: {candidates}. "
        "Check Appium base path (root vs /wd/hub) and that uiautomator2 driver is installed "
        "(appium driver install uiautomator2)."
    )

def get_window_size(driver):
    s = driver.get_window_size()
    return int(s["width"]), int(s["height"])

def capture_bgr(driver):
    png = driver.get_screenshot_as_png()
    arr = np.frombuffer(png, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
