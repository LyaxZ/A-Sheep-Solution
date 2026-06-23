"""
羊了个羊 自动通关 - 配置文件
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---- 项目路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATE_DIR = DATA_DIR / "templates"
SCREENSHOT_DIR = DATA_DIR / "screenshots"

# ---- 微信窗口 ----
WECHAT_WINDOW_TITLE = "微信"          # 微信窗口标题（用于自动定位）
WECHAT_WINDOW_SIZE = (400, 720)       # 小程序窗口大致尺寸 (宽, 高)

# ---- 游戏区域（在微信窗口内的相对坐标，需根据实际情况微调）----
# 这些值后面可以通过校准工具自动确定
GAME_BOARD_REGION = (20, 180, 380, 520)   # (x1, y1, x2, y2) 棋盘区域
BUFFER_REGION = (20, 540, 380, 600)       # 底部缓冲槽区域

# ---- 方块参数 ----
TILE_SIZE = (50, 50)          # 单个方块的宽高 (像素)
TILE_MATCH_THRESHOLD = 0.75   # 模板匹配阈值 (0~1), 越高越严格

# ---- 游戏规则 ----
BUFFER_MAX = 7       # 缓冲槽容量
MATCH_COUNT = 3      # 消除需要的数量
LAYER_COUNT = 4      # 最大层数

# ---- 模板图标类型 ----
# 羊了个羊常见图标类型名称（用于文件名）
TILE_TYPES = [
    "wood", "hay", "water", "fire", "glove",
    "fork", "knife", "ball", "corn", "carrot",
    "cabbage", "milk", "brush", "scissors", "bell"
]

# ---- 道具 ----
@dataclass
class Item:
    name: str                   # 中文名
    key: str                    # 英文标识
    region: tuple               # 道具按钮在窗口中的区域 (x1,y1,x2,y2)
    description: str
    is_once: bool = False       # 是否为一次性道具（如复活只能用一次）
    is_emergency: bool = False  # 是否为紧急触发（槽满时自动弹出）


ITEMS = {
    "shuffle": Item(
        name="洗牌",
        key="shuffle",
        region=(20, 620, 100, 660),
        description="重新排列场上所有方块"
    ),
    "undo": Item(
        name="撤销",
        key="undo",
        region=(120, 620, 200, 660),
        description="撤销上一步操作"
    ),
    "move_out": Item(
        name="移出",
        key="move_out",
        region=(220, 620, 300, 660),
        description="将缓冲槽中3个方块移到备用区"
    ),
    "revive": Item(
        name="复活",
        key="revive",
        region=(100, 400, 300, 480),   # 复活按钮通常在弹窗中间偏下
        description="槽满后移除最左侧3个方块，每局仅一次",
        is_once=True,
        is_emergency=True,
    ),
}

# 普通道具（随时可用）
NORMAL_ITEMS = ["shuffle", "undo", "move_out"]
# 紧急道具（触发条件出现时才可用）
EMERGENCY_ITEMS = ["revive"]

# ---- 死胡同时的道具推荐逻辑 ----
def recommend_item(buffer_types: list, blocked_stats: dict,
                   revive_used: bool = False) -> str | None:
    """
    根据当前局面推荐道具，返回道具 key 或 None
    
    关键规则：槽满时洗牌/撤销/移出不可用，仅复活有效！
    
    buffer_types: 缓冲槽中的方块类型列表
    blocked_stats: 每种类型被阻塞的方块数量
    revive_used: 本局是否已使用过复活
    """
    from collections import Counter

    buffer_full = len(buffer_types) >= BUFFER_MAX

    # === 槽满情况：仅复活可用 ===
    if buffer_full:
        if not revive_used:
            counts = Counter(buffer_types)
            if not any(c >= MATCH_COUNT for c in counts.values()):
                return "revive"
        return None  # 槽满 + 复活已用 = 无解

    # === 槽未满 ===
    unique = set(buffer_types)

    # 规则1：类型分散（≥5种）→ 建议移出
    if len(unique) >= 5:
        return "move_out"

    # 规则2：缓冲中有即将成对的但都被压住 → 建议洗牌
    counts = Counter(buffer_types)
    for t, c in counts.items():
        if c >= 2 and blocked_stats.get(t, 0) > 0:
            return "shuffle"

    # 规则3：缓冲槽接近满（≥6）→ 建议移出
    if len(buffer_types) >= 6:
        return "move_out"

    # 规则4：大量方块被阻塞 → 建议洗牌
    total_blocked = sum(blocked_stats.values())
    if total_blocked > 20:
        return "shuffle"

    return None
