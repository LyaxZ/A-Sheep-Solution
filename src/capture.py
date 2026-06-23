"""
屏幕截图模块 - 比例坐标 + 颜色自动校准，适配任意窗口大小
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss

from config import (
    WECHAT_WINDOW_TITLE,
    GAME_BOARD_RATIO,
    BUFFER_RATIO,
    SCREENSHOT_DIR,
    WindowInfo,
    auto_detect_regions,
    apply_auto_regions,
)


def find_wechat_window() -> gw.Window | None:
    """查找小程序窗口"""
    for win in gw.getWindowsWithTitle(WECHAT_WINDOW_TITLE):
        if win.visible and win.width > 200 and win.height > 200:
            return win
    return None


def get_window_info(win: gw.Window | None = None) -> WindowInfo | None:
    """获取窗口信息对象"""
    if win is None:
        win = find_wechat_window()
    if win is None:
        return None
    return WindowInfo(win.left, win.top, win.width, win.height)


def get_game_screenshot(win_info: WindowInfo | None = None) -> np.ndarray | None:
    """截取小程序窗口完整截图，返回 BGR。
    同时修正 win_info 的宽高为截图实际尺寸（处理 DPI 缩放）"""
    if win_info is None:
        win_info = get_window_info()
    if win_info is None:
        print("[capture] 未找到小程序窗口")
        return None

    monitor = {"left": win_info.left, "top": win_info.top,
               "width": win_info.width, "height": win_info.height}
    with mss() as sct:
        img = sct.grab(monitor)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    # 用截图实际尺寸修正 win_info（处理 DPI 缩放差异）
    h, w = frame.shape[:2]
    if w != win_info.width or h != win_info.height:
        print(f"  [capture] DPI缩放: 窗口{win_info.width}x{win_info.height} → 截图{w}x{h}")
        win_info.width = w
        win_info.height = h

    return frame


def crop_region(frame: np.ndarray, ratio: tuple) -> np.ndarray:
    """按比例裁剪区域"""
    h, w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = ratio
    x1, y1 = max(0, int(w * rx1)), max(0, int(h * ry1))
    x2, y2 = min(w, int(w * rx2)), min(h, int(h * ry2))
    return frame[y1:y2, x1:x2]


def crop_game_board(frame: np.ndarray) -> np.ndarray:
    return crop_region(frame, GAME_BOARD_RATIO)


def crop_buffer(frame: np.ndarray) -> np.ndarray:
    return crop_region(frame, BUFFER_RATIO)


def save_debug_screenshot(frame: np.ndarray, name: str = "debug"):
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = SCREENSHOT_DIR / f"{name}_{ts}.png"
    cv2.imwrite(str(filename), frame)
    print(f"  [capture] 截图: {filename}")
    return filename


def auto_calibrate() -> bool:
    """自动校准：截图 → 颜色检测 → 更新全局比例。成功返回 True。"""
    print("[calibrate] 自动校准中...")
    win_info = get_window_info()
    if win_info is None:
        print(f"[calibrate] FAIL: window not found")
        return False

    print(f"[calibrate] 窗口: {win_info.width}x{win_info.height}")

    frame = get_game_screenshot(win_info)
    if frame is None:
        return False

    detected = auto_detect_regions(frame)
    if not detected or "board" not in detected:
        print("[calibrate] FAIL: using default ratios")
        save_debug_screenshot(frame, "calibrate_failed")
        # 也保存当前裁剪的棋盘，方便检查
        board = crop_game_board(frame)
        save_debug_screenshot(board, "calibrate_board_default")
        return True

    apply_auto_regions(detected)
    print(f"[calibrate] OK board: {GAME_BOARD_RATIO}")
    print(f"[calibrate] OK buffer: {BUFFER_RATIO}")

    board = crop_game_board(frame)
    save_debug_screenshot(board, "calibrate_board")
    return True
