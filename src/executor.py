"""
动作执行模块 - 比例坐标 + 自动校准，适配任意窗口位置/大小
"""

import time
import random

import pyautogui
import pygetwindow as gw

from config import (
    ITEMS,
    GAME_BOARD_RATIO,
    WECHAT_WINDOW_TITLE,
    WindowInfo,
)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03


class ClickExecutor:
    """模拟点击执行器，使用比例坐标"""

    def __init__(self):
        self.win_info: WindowInfo | None = None
        self.click_delay = (0.06, 0.12)

    def calibrate(self) -> bool:
        """定位窗口，返回是否成功"""
        for w in gw.getWindowsWithTitle(WECHAT_WINDOW_TITLE):
            if w.visible and w.width > 200 and w.height > 200:
                self.win_info = WindowInfo(w.left, w.top, w.width, w.height)
                print(f"[executor] 窗口: {w.width}x{w.height} @({w.left},{w.top})")
                return True

        print(f"[executor] ✗ 未找到窗口 [{WECHAT_WINDOW_TITLE}]")
        return False

    def click_tile(self, tile_bbox: tuple):
        """
        点击方块。
        tile_bbox: (x, y, w, h) 在棋盘内的像素坐标
        """
        x, y, w, h = tile_bbox
        # 棋盘区域在窗口内的像素偏移
        bx, by, _, _ = self.win_info.ratio_to_region(GAME_BOARD_RATIO)

        # 屏幕绝对坐标
        sx = self.win_info.client_left + bx + x + w // 2
        sy = self.win_info.client_top + by + y + h // 2
        self._click(sx, sy)

    def click_item(self, item_key: str):
        """点击道具按钮"""
        item = ITEMS.get(item_key)
        if item is None:
            return
        sx, sy = self.win_info.ratio_to_screen_center(item.ratio)
        print(f"[executor] 道具: {item.name}")
        self._click(sx, sy)

    def click_restart(self):
        """点击重新开始（窗口中央偏下）"""
        if self.win_info:
            sx = self.win_info.client_left + self.win_info.width // 2
            sy = self.win_info.client_top + int(self.win_info.height * 0.72)
            self._click(sx, sy)

    def _click(self, x: int, y: int):
        jx = random.randint(-2, 2)
        jy = random.randint(-2, 2)
        pyautogui.click(x + jx, y + jy)
        time.sleep(random.uniform(*self.click_delay))


def user_item_prompt(recommended_key: str | None,
                     revive_used: bool = False,
                     buffer_full: bool = False) -> str | None:
    """弹出道具选择提示，验证可用性"""
    if recommended_key:
        item = ITEMS.get(recommended_key)
        desc = f"{item.name} - {item.description}" if item else recommended_key
        prompt = (f"\n{'='*50}\n"
                  f"⚠ 自动求解失败，推荐: [{recommended_key}] {desc}\n\n"
                  f"可选道具:\n")
    else:
        prompt = (f"\n{'='*50}\n"
                  f"⚠ 自动求解失败\n可选道具:\n")

    for key, item in ITEMS.items():
        if key == "revive":
            if revive_used:
                prompt += f"  [{key}] {item.name} (已用)\n"
            elif buffer_full:
                prompt += f"  [{key}] {item.name} ← 唯一可用!\n"
        elif buffer_full:
            prompt += f"  [{key}] {item.name} (槽满不可用)\n"
        else:
            prompt += f"  [{key}] {item.name}\n"

    prompt += "  [skip] 跳过  [quit] 退出\n选择: "

    while True:
        c = input(prompt).strip().lower()
        if c in ("skip", ""):
            return None
        if c == "quit":
            return "quit"

        if c == "revive":
            if revive_used:
                print("❌ 复活已用过")
                continue
            if not buffer_full:
                print("❌ 复活仅槽满时可用")
                continue
            return c

        if c in ITEMS:
            if buffer_full:
                print(f"❌ 槽满时 [{c}] 不可用")
                continue
            return c

        print(f"无效: {c}")
