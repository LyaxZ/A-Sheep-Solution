"""
方块检测模块 - 识别棋盘上的所有方块：位置、类型、是否可点击
"""

from pathlib import Path

import cv2
import numpy as np

from config import (
    TEMPLATE_DIR,
    TILE_SIZE,
    TILE_MATCH_THRESHOLD,
    TILE_TYPES,
)


# ---- 数据结构 ----

class Tile:
    """表示一个方块"""
    __slots__ = ("id", "tile_type", "bbox", "center", "area", "free", "layer")

    def __init__(self, tile_id: int, tile_type: str, bbox: tuple, layer: int = 0):
        """
        bbox: (x, y, w, h) 方块边界框
        """
        self.id = tile_id
        self.tile_type = tile_type
        self.bbox = bbox          # (x, y, w, h)
        x, y, w, h = bbox
        self.center = (x + w // 2, y + h // 2)
        self.area = w * h
        self.free = True           # 是否可点击（不被上层方块压住）
        self.layer = layer         # 所在层级（越大越靠上）

    def __repr__(self):
        status = "✓" if self.free else "✗"
        return f"Tile#{self.id}[{self.tile_type}]@({self.bbox[0]},{self.bbox[1]}){status}"

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.tile_type,
            "bbox": self.bbox,
            "center": self.center,
            "free": self.free,
            "layer": self.layer,
        }


# ---- 模板管理 ----

class TemplateMatcher:
    """管理图标模板的加载与匹配"""

    def __init__(self):
        self.templates: dict[str, list[np.ndarray]] = {}  # type -> [template_images]
        self._load_templates()

    def _load_templates(self):
        """从 TEMPLATE_DIR 加载所有模板图片"""
        if not TEMPLATE_DIR.exists():
            print(f"[detector] 模板目录不存在: {TEMPLATE_DIR}")
            print(f"[detector] 请先运行校准模式收集方块图标模板")
            return

        for img_path in TEMPLATE_DIR.glob("*.png"):
            stem = img_path.stem.lower()
            # 文件名格式: {type}_{序号}.png 或 {type}.png
            t_type = stem.rsplit("_", 1)[0] if "_" in stem else stem
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, TILE_SIZE)
            if t_type not in self.templates:
                self.templates[t_type] = []
            self.templates[t_type].append(img)

        if self.templates:
            print(f"[detector] 已加载 {sum(len(v) for v in self.templates.values())} 个模板, "
                  f"{len(self.templates)} 种类型")
        else:
            print("[detector] 警告: 未加载任何模板！")

    def classify(self, tile_img: np.ndarray) -> tuple[str, float]:
        """
        对单个方块图像进行分类
        返回 (类型名, 置信度)
        """
        best_type = "unknown"
        best_score = 0.0

        tile_img = cv2.resize(tile_img, TILE_SIZE)

        for t_type, templates in self.templates.items():
            for tmpl in templates:
                # 使用归一化互相关系数
                result = cv2.matchTemplate(tile_img, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_type = t_type

        return best_type, best_score

    @property
    def type_names(self) -> list[str]:
        return list(self.templates.keys())


# ---- 方块检测器 ----

class TileDetector:
    """检测棋盘上的所有方块"""

    def __init__(self):
        self.matcher = TemplateMatcher()
        self.tile_counter = 0
        self.expected_tile_area = TILE_SIZE[0] * TILE_SIZE[1]

    def detect(self, board_img: np.ndarray) -> list[Tile]:
        """
        主检测流程：找方块 → 分类 → 判断可点击性
        board_img: BGR 棋盘区域图像
        """
        self.tile_counter = 0
        tiles = []

        # 步骤1：查找所有可能是方块的矩形区域
        candidates = self._find_tile_candidates(board_img)
        if not candidates:
            print("[detector] 未检测到任何方块")
            return tiles

        print(f"[detector] 检测到 {len(candidates)} 个候选方块")

        # 步骤2：对每个候选方块进行模板匹配分类
        for bbox in candidates:
            x, y, w, h = bbox
            # 提取方块中心区域（图标部分，缩小以避免边框干扰）
            margin = max(4, w // 8)
            icon_region = board_img[y + margin:y + h - margin,
                                    x + margin:x + w - margin]

            if icon_region.size == 0:
                continue

            tile_type, score = self.matcher.classify(icon_region)

            if score >= TILE_MATCH_THRESHOLD or self.matcher.type_names == []:
                # 如果没有模板，全部标记为 unknown
                if not self.matcher.type_names:
                    tile_type = "unknown"
                tile = Tile(self.tile_counter, tile_type, bbox)
                tiles.append(tile)
                self.tile_counter += 1

        print(f"[detector] 成功分类 {len(tiles)} 个方块")

        # 步骤3：计算层级并判断可点击性
        self._compute_layers(tiles)

        return tiles

    def _find_tile_candidates(self, board_img: np.ndarray) -> list[tuple]:
        """
        查找所有方块的候选边界框
        使用边缘检测 + 轮廓查找
        """
        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny 边缘检测
        edges = cv2.Canny(blurred, 30, 100)

        # 膨胀连接断边
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        min_area = self.expected_tile_area * 0.25   # 最小面积（考虑遮挡）
        max_area = self.expected_tile_area * 1.8

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # 面积过滤
            if min_area <= area <= max_area:
                # 宽高比过滤（方块接近正方形，容忍一定变形）
                aspect = w / h if h > 0 else 0
                if 0.5 <= aspect <= 2.0:
                    candidates.append((x, y, w, h))

        # 合并重叠度过高的候选框（去重）
        candidates = self._merge_overlapping(candidates)
        return candidates

    def _merge_overlapping(self, boxes: list[tuple],
                            iou_threshold: float = 0.6) -> list[tuple]:
        """合并 IoU 过高的重叠框"""
        if len(boxes) <= 1:
            return boxes

        # 按面积降序排列
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep = []

        for box in boxes:
            overlapped = False
            for kept in keep:
                if self._compute_iou(box, kept) > iou_threshold:
                    overlapped = True
                    break
            if not overlapped:
                keep.append(box)

        return keep

    def _compute_iou(self, a: tuple, b: tuple) -> float:
        """计算两个边界框的 IoU"""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax + aw, bx + bw)
        inter_y2 = min(ay + ah, by + bh)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = (aw * ah) + (bw * bh) - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _compute_layers(self, tiles: list[Tile]):
        """
        计算每个方块的层级和可点击性
        规则：如果方块 A 的中心点落在方块 B 的区域内，且 B 面积更大，
        则 B 在 A 上层，A 被 B 遮挡，A 不可点击
        """
        if not tiles:
            return

        n = len(tiles)

        # 按面积排序计算层级：面积越大越靠上（上层方块更完整）
        sorted_tiles = sorted(tiles, key=lambda t: t.area, reverse=True)
        for i, tile in enumerate(sorted_tiles):
            tile.layer = n - i  # 层级越高越靠上

        # 判断每个方块是否被其他方块遮挡
        for tile in tiles:
            cx, cy = tile.center
            for other in tiles:
                if other.id == tile.id:
                    continue
                ox, oy, ow, oh = other.bbox
                # 检查 tile 的中心是否在 other 的区域内
                if (ox <= cx <= ox + ow) and (oy <= cy <= oy + oh):
                    # other 在 tile 上层 → tile 被遮挡
                    if other.layer > tile.layer:
                        tile.free = False
                        break

        free_count = sum(1 for t in tiles if t.free)
        print(f"[detector] 可点击方块: {free_count}/{len(tiles)}")


# ---- 辅助函数 ----

def get_free_tiles(tiles: list[Tile]) -> list[Tile]:
    """获取所有可点击的方块"""
    return [t for t in tiles if t.free]


def get_tiles_by_type(tiles: list[Tile], tile_type: str) -> list[Tile]:
    """按类型筛选方块"""
    return [t for t in tiles if t.tile_type == tile_type]


def count_by_type(tiles: list[Tile]) -> dict[str, int]:
    """统计各类方块数量"""
    from collections import Counter
    return dict(Counter(t.tile_type for t in tiles))


if __name__ == "__main__":
    # 测试：读取截图并检测
    from capture import find_wechat_window, get_game_screenshot, crop_game_board

    win = find_wechat_window()
    if win:
        frame = get_game_screenshot(win)
        if frame is not None:
            board = crop_game_board(frame)
            detector = TileDetector()
            tiles = detector.detect(board)
            print(f"\n总计检测到 {len(tiles)} 个方块:")
            for t in tiles:
                print(f"  {t}")
            print(f"\n可点击: {len(get_free_tiles(tiles))}")
            print(f"类型分布: {count_by_type(tiles)}")
