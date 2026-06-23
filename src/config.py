"""
羊了个羊 自动通关 - 配置文件
所有坐标为窗口比例 (0~1)，自动适配任意窗口大小
"""

from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

# ---- 项目路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
SCREENSHOT_DIR = DATA_DIR / "screenshots"

# ---- 小程序窗口（自动检测）----
WECHAT_WINDOW_TITLE = "羊了个羊：星球"

# ---- 游戏区域比例（窗口内的比例位置，0~1）----
GAME_BOARD_RATIO = (0.05, 0.22, 0.95, 0.67)    # 棋盘区域
BUFFER_RATIO = (0.05, 0.67, 0.95, 0.76)          # 缓冲槽区域
# 道具按钮（窗口底部3个蓝按钮）
ITEMS_RATIO = {
    "shuffle":   (0.05, 0.80, 0.30, 0.92),
    "undo":      (0.33, 0.80, 0.58, 0.92),
    "move_out":  (0.62, 0.80, 0.87, 0.92),
}
REVIVE_RATIO = (0.25, 0.52, 0.75, 0.62)          # 复活弹窗按钮

# ---- 颜色阈值（自动检测用）----
BOARD_GREEN_LOW  = (25, 25, 30)     # 棋盘绿色 HSV 下限 (放宽青绿色)
BOARD_GREEN_HIGH = (100, 255, 255)  # 棋盘绿色 HSV 上限
ITEM_BLUE_LOW    = (90, 40, 30)     # 道具蓝色 HSV 下限
ITEM_BLUE_HIGH   = (135, 255, 255)  # 道具蓝色 HSV 上限

# ---- 方块参数 ----
TILE_SIZE = (50, 50)
TILE_MATCH_THRESHOLD = 0.45

# ---- 游戏规则 ----
BUFFER_MAX = 7
MATCH_COUNT = 3

# ---- 道具定义 ----
@dataclass
class Item:
    name: str
    key: str
    ratio: tuple
    description: str
    is_once: bool = False
    is_emergency: bool = False

ITEMS = {
    "shuffle":   Item("洗牌", "shuffle", ITEMS_RATIO["shuffle"], "重新排列场上所有方块"),
    "undo":      Item("撤销", "undo", ITEMS_RATIO["undo"], "撤销上一步操作"),
    "move_out":  Item("移出", "move_out", ITEMS_RATIO["move_out"], "将缓冲槽中3个方块移到备用区"),
    "revive":    Item("复活", "revive", REVIVE_RATIO, "槽满后移除最左侧3个方块，每局仅一次",
                      is_once=True, is_emergency=True),
}


# ---- 窗口信息与坐标转换 ----

@dataclass
class WindowInfo:
    """Window size/position + ratio-to-pixel conversion"""
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def client_left(self) -> int:
        return self.left + 8

    @property
    def client_top(self) -> int:
        return self.top + 31

    def ratio_to_region(self, ratio: tuple) -> tuple:
        """Ratio -> pixel region in window (x1,y1,x2,y2)"""
        rx1, ry1, rx2, ry2 = ratio
        return (int(self.width * rx1), int(self.height * ry1),
                int(self.width * rx2), int(self.height * ry2))

    def ratio_to_screen_center(self, ratio: tuple) -> tuple:
        """Ratio -> absolute screen coordinates (cx, cy)"""
        px1, py1, px2, py2 = self.ratio_to_region(ratio)
        return (self.client_left + (px1 + px2) // 2,
                self.client_top + (py1 + py2) // 2)

    def board_region(self) -> tuple:
        return self.ratio_to_region(GAME_BOARD_RATIO)

    def buffer_region(self) -> tuple:
        return self.ratio_to_region(BUFFER_RATIO)


# ---- 自动检测游戏区域 ----

def auto_detect_regions(frame_bgr: np.ndarray) -> dict:
    """
    基于颜色自动检测棋盘、缓冲槽、道具按钮。
    frame_bgr: 窗口完整截图 (BGR)
    返回 {'board': 比例, 'buffer': 比例, 'item_shuffle': 比例, ...}
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = frame_bgr.shape[:2]
    result = {}

    # 1. 绿色棋盘
    green = cv2.inRange(hsv, BOARD_GREEN_LOW, BOARD_GREEN_HIGH)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if bw * bh > w * h * 0.1:
            result["board"] = (x / w, y / h, (x + bw) / w, (y + bh) / h)

    # 2. 缓冲槽：棋盘正下方
    if "board" in result:
        bx1, by1, bx2, by2 = result["board"]
        result["buffer"] = (bx1 + 0.02, by2 + 0.005, bx2 - 0.02, min(by2 + 0.10, 1.0))
    else:
        result["buffer"] = BUFFER_RATIO

    # 3. 蓝色道具按钮
    blue = cv2.inRange(hsv, ITEM_BLUE_LOW, ITEM_BLUE_HIGH)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_boxes = []
    for cnt in contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw > 20 and rh > 20 and ry / h > 0.70:
            blue_boxes.append((rx / w, ry / h, (rx + rw) / w, (ry + rh) / h))

    if len(blue_boxes) >= 3:
        blue_boxes.sort(key=lambda b: b[0])
        for i, key in enumerate(["shuffle", "undo", "move_out"]):
            result[f"item_{key}"] = blue_boxes[i]
        print(f"  [auto-detect] 道具: {len(blue_boxes)} 个按钮")
    else:
        print(f"  [auto-detect] 道具: 仅 {len(blue_boxes)} 个, 用默认比例")

    # 4. 复活弹窗按钮
    if "board" in result:
        bx1, by1, bx2, by2 = result["board"]
        result["revive"] = (bx1 + 0.15, by1 + 0.40, bx2 - 0.15, by1 + 0.55)
    else:
        result["revive"] = REVIVE_RATIO

    return result


def apply_auto_regions(detected: dict):
    """将自动检测结果写入全局比例"""
    global GAME_BOARD_RATIO, BUFFER_RATIO, ITEMS_RATIO
    if "board" in detected:
        GAME_BOARD_RATIO = detected["board"]
    if "buffer" in detected:
        BUFFER_RATIO = detected["buffer"]
    for key in ["shuffle", "undo", "move_out"]:
        if f"item_{key}" in detected:
            ITEMS_RATIO[key] = detected[f"item_{key}"]
    if "revive" in detected:
        ITEMS["revive"].ratio = detected["revive"]


# ---- 道具推荐逻辑 ----

def recommend_item(buffer_types: list, blocked_stats: dict,
                   revive_used: bool = False) -> str | None:
    """根据局面推荐道具。槽满时仅复活可用。"""
    from collections import Counter

    if len(buffer_types) >= BUFFER_MAX:
        if not revive_used:
            counts = Counter(buffer_types)
            if not any(c >= MATCH_COUNT for c in counts.values()):
                return "revive"
        return None

    unique = set(buffer_types)
    if len(unique) >= 5:
        return "move_out"

    counts = Counter(buffer_types)
    for t, c in counts.items():
        if c >= 2 and blocked_stats.get(t, 0) > 0:
            return "shuffle"

    if len(buffer_types) >= 6:
        return "move_out"
    if sum(blocked_stats.values()) > 20:
        return "shuffle"

    return None
