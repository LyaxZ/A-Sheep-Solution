"""
屏幕截图模块 - 定位微信小程序窗口并截取游戏画面
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pygetwindow as gw
from mss import mss

from config import (
    WECHAT_WINDOW_TITLE,
    GAME_BOARD_REGION,
    BUFFER_REGION,
    SCREENSHOT_DIR,
)


def find_wechat_window() -> gw.Window | None:
    """查找微信窗口"""
    windows = gw.getWindowsWithTitle(WECHAT_WINDOW_TITLE)
    if not windows:
        return None
    # 如果有多个，取第一个可见的
    for win in windows:
        if win.visible and win.width > 200 and win.height > 200:
            return win
    return windows[0] if windows else None


def get_game_screenshot(win: gw.Window | None = None) -> np.ndarray | None:
    """
    截取微信窗口的完整截图
    返回 BGR 格式的 numpy 数组
    """
    if win is None:
        win = find_wechat_window()
    if win is None:
        print("[capture] 未找到微信窗口，请确保微信已打开且小程序在前台")
        return None

    monitor = {
        "left": win.left,
        "top": win.top,
        "width": win.width,
        "height": win.height,
    }

    with mss() as sct:
        img = sct.grab(monitor)
        # mss 返回 BGRA，转为 BGR
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame


def crop_game_board(frame: np.ndarray) -> np.ndarray:
    """从完整截图中裁剪游戏棋盘区域"""
    x1, y1, x2, y2 = GAME_BOARD_REGION
    h, w = frame.shape[:2]
    # 边界保护
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def crop_buffer(frame: np.ndarray) -> np.ndarray:
    """从完整截图中裁剪缓冲槽区域"""
    x1, y1, x2, y2 = BUFFER_REGION
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def get_window_client_offset(win: gw.Window) -> tuple[int, int]:
    """
    获取窗口客户区相对于窗口左上角的偏移
    用于将客户区坐标转换为屏幕绝对坐标
    """
    # pygetwindow 给出的 left/top 是窗口外框位置
    # 对于大多数 Windows 应用，客户区偏移约为 (8, 31)（标题栏+边框）
    # 这里返回一个估计值，可后续校准
    return (win.left + 8, win.top + 31)


def save_debug_screenshot(frame: np.ndarray, name: str = "debug"):
    """保存调试截图"""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = SCREENSHOT_DIR / f"{name}_{ts}.png"
    cv2.imwrite(str(filename), frame)
    print(f"[capture] 截图已保存: {filename}")
    return filename


if __name__ == "__main__":
    # 测试：截图并保存
    win = find_wechat_window()
    if win:
        print(f"找到窗口: {win.title}, 位置: ({win.left},{win.top}), 大小: ({win.width},{win.height})")
        frame = get_game_screenshot(win)
        if frame is not None:
            save_debug_screenshot(frame, "full")
            board = crop_game_board(frame)
            save_debug_screenshot(board, "board")
    else:
        print("未找到微信窗口")
