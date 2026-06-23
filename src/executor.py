"""
动作执行模块 - 在微信窗口中模拟鼠标点击
"""

import time
import random

import pyautogui
import pygetwindow as gw

from config import GAME_BOARD_REGION, ITEMS

# 安全设置：移动到角落时不会触发异常
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05  # 每次操作后的暂停时间


class ClickExecutor:
    """
    模拟点击执行器
    负责将棋盘坐标转换为屏幕绝对坐标并点击
    """

    def __init__(self):
        self.win: gw.Window | None = None
        self.offset_x = 0
        self.offset_y = 0
        self.click_delay = (0.08, 0.15)  # 点击间隔随机范围（秒）

    def calibrate(self):
        """定位微信窗口并计算坐标偏移"""
        windows = gw.getWindowsWithTitle("微信")
        for w in windows:
            if w.visible and w.width > 200 and w.height > 200:
                self.win = w
                break

        if self.win is None:
            print("[executor] 警告: 未找到微信窗口")
            return False

        # Windows 客户区偏移（标题栏 + 边框）
        # pygetwindow 获取的是窗口外框坐标
        self.offset_x = self.win.left + 8
        self.offset_y = self.win.top + 31

        print(f"[executor] 校准完成: 窗口 [{self.win.title}] "
              f"({self.win.width}x{self.win.height}) "
              f"偏移=({self.offset_x}, {self.offset_y})")
        return True

    def click_tile(self, tile_bbox: tuple):
        """
        点击方块
        tile_bbox: (x, y, w, h) 在棋盘区域内的相对坐标
        """
        x, y, w, h = tile_bbox
        board_x1, board_y1, _, _ = GAME_BOARD_REGION

        # 计算方块的屏幕绝对坐标（点方块中心）
        screen_x = self.offset_x + board_x1 + x + w // 2
        screen_y = self.offset_y + board_y1 + y + h // 2

        self._click(screen_x, screen_y)

    def click_item(self, item_key: str):
        """点击道具按钮"""
        item = ITEMS.get(item_key)
        if item is None:
            print(f"[executor] 未知道具: {item_key}")
            return

        x1, y1, x2, y2 = item.region
        screen_x = self.offset_x + (x1 + x2) // 2
        screen_y = self.offset_y + (y1 + y2) // 2

        print(f"[executor] 使用道具: {item.name}")
        self._click(screen_x, screen_y)

    def click_restart(self):
        """点击重新开始按钮（通常在失败弹窗上）"""
        # 屏幕中央偏下
        if self.win:
            screen_x = self.offset_x + self.win.width // 2
            screen_y = self.offset_y + int(self.win.height * 0.7)
            self._click(screen_x, screen_y)

    # ---- 内部方法 ----

    def _click(self, x: int, y: int):
        """在指定屏幕坐标点击"""
        # 添加微小随机偏移，模拟人类点击
        jitter_x = random.randint(-2, 2)
        jitter_y = random.randint(-2, 2)

        pyautogui.click(x + jitter_x, y + jitter_y)

        # 随机延迟，避免点击过快被检测
        delay = random.uniform(*self.click_delay)
        time.sleep(delay)

    def move_to_safe(self):
        """将鼠标移到安全位置（避免干扰视觉识别）"""
        if self.win:
            safe_x = self.offset_x + self.win.width + 50
            safe_y = self.offset_y + self.win.height + 50
            pyautogui.moveTo(safe_x, safe_y, duration=0.1)


def user_item_prompt(recommended_key: str | None,
                     revive_used: bool = False,
                     buffer_full: bool = False) -> str | None:
    """
    弹出道具使用提示，等待用户选择
    recommended_key: 推荐的道具 key
    revive_used: 本局是否已用过复活
    buffer_full: 缓冲槽是否已满（触发复活条件）
    返回用户选择的道具 key，或 None 表示不使用
    """
    if recommended_key:
        item = ITEMS.get(recommended_key)
        item_desc = f"{item.name} - {item.description}" if item else recommended_key
        prompt_text = (
            f"\n{'='*50}\n"
            f"⚠️ 自动求解失败，建议使用道具:\n"
            f"   [{recommended_key}] {item_desc}\n\n"
            f"可选道具:\n"
        )
    else:
        prompt_text = (
            f"\n{'='*50}\n"
            f"⚠️ 自动求解失败，无推荐道具\n"
            f"可选道具:\n"
        )

    # 显示可用道具列表
    for key, item in ITEMS.items():
        if key == "revive":
            # 复活是紧急道具，只在槽满时可用
            if revive_used:
                prompt_text += f"  [{key}] {item.name} (已用) - {item.description}\n"
            elif buffer_full:
                prompt_text += f"  [{key}] {item.name} ⬅ 唯一可用! - {item.description}\n"
            else:
                # 槽未满时复活不可用，不显示
                pass
        else:
            # 普通道具：槽满时不可用
            if buffer_full:
                prompt_text += f"  [{key}] {item.name} (槽满不可用) - {item.description}\n"
            else:
                prompt_text += f"  [{key}] {item.name} - {item.description}\n"

    prompt_text += (
        f"  [skip] 不使用道具\n"
        f"  [quit] 退出程序\n"
        f"请输入选择: "
    )

    while True:
        choice = input(prompt_text).strip().lower()

        if choice == "skip" or choice == "":
            return None
        if choice == "quit":
            return "quit"

        # 验证道具可用性
        if choice == "revive":
            if revive_used:
                print(f"❌ 复活已使用过，本局无法再次使用")
                continue
            if not buffer_full:
                print(f"❌ 复活仅在缓冲槽满时可用")
                continue
            return choice

        if choice in ITEMS:
            if buffer_full:
                print(f"❌ 槽满时 [{choice}] 不可用，仅复活可选")
                continue
            return choice

        print(f"无效输入: {choice}，请重新输入")


if __name__ == "__main__":
    executor = ClickExecutor()
    if executor.calibrate():
        print("校准成功，3秒后点击屏幕中心...")
        time.sleep(3)
        executor._click(executor.offset_x + 200, executor.offset_y + 400)
        print("点击完成")
