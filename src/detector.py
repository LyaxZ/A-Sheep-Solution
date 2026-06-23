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
)


# ---- 数据结构 ----

class Tile:
    """表示一个方块，含可见比例"""
    __slots__ = ("id", "tile_type", "bbox", "center", "area",
                 "free", "layer", "visible_ratio", "match_score")

    def __init__(self, tile_id: int, tile_type: str, bbox: tuple, layer: int = 0):
        self.id = tile_id
        self.tile_type = tile_type
        self.bbox = bbox          # (x, y, w, h)
        x, y, w, h = bbox
        self.center = (x + w // 2, y + h // 2)
        self.area = w * h
        self.free = True
        self.layer = layer
        self.visible_ratio = 1.0
        self.match_score = 0.0    # 模板匹配分（越高越完整→上层）

    def __repr__(self):
        status = "OK" if self.free else "XX"
        return f"Tile#{self.id}[{self.tile_type}]@({self.bbox[0]},{self.bbox[1]}) {self.visible_ratio:.0%}{status}"

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
        混合分类：模板匹配 + 直方图，取两者最高分。
        """
        best_type = "unknown"
        best_tmpl_score = 0.0

        tile_img = cv2.resize(tile_img, TILE_SIZE)

        # 模板匹配
        for t_type, templates in self.templates.items():
            for tmpl in templates:
                result = cv2.matchTemplate(tile_img, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_tmpl_score:
                    best_tmpl_score = max_val
                    best_type = t_type

        # 直方图匹配
        h_type, h_score = self._classify_by_histogram(tile_img)

        # 取两者中高分（直方图对颜色敏感，模板对形状敏感）
        if h_score > best_tmpl_score and h_score > 0.5:
            return h_type, h_score

        return best_type, best_tmpl_score

    def _classify_by_histogram(self, tile_img: np.ndarray) -> tuple[str, float]:
        """颜色直方图相似度匹配（回退方案）"""
        hsv = cv2.cvtColor(tile_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        best_type = "unknown"
        best_score = 0.0

        for t_type, templates in self.templates.items():
            for tmpl in templates:
                tmpl_hsv = cv2.cvtColor(tmpl, cv2.COLOR_BGR2HSV)
                tmpl_hist = cv2.calcHist([tmpl_hsv], [0, 1], None,
                                          [32, 32], [0, 180, 0, 256])
                cv2.normalize(tmpl_hist, tmpl_hist, 0, 1, cv2.NORM_MINMAX)

                score = cv2.compareHist(hist, tmpl_hist, cv2.HISTCMP_CORREL)
                if score > best_score:
                    best_score = score
                    best_type = t_type

        return best_type, best_score

    @property
    def type_names(self) -> list[str]:
        return list(self.templates.keys())


# ---- 方块检测器 ----

class TileDetector:
    """检测棋盘上的所有方块 + 缓冲槽"""

    def __init__(self):
        self.matcher = TemplateMatcher()
        self.tile_counter = 0

    def detect_buffer(self, buffer_img: np.ndarray) -> list[str]:
        """
        检测缓冲槽中的方块类型。
        buffer_img: 缓冲槽区域图像 (BGR)
        返回类型名列表，空位不计入
        """
        if buffer_img is None or buffer_img.size == 0:
            return []

        h, w = buffer_img.shape[:2]
        # 缓冲槽有 7 个格子，水平排列
        slot_w = w / 7
        buffer_types = []

        for i in range(7):
            x1 = int(i * slot_w) + 2
            x2 = int((i + 1) * slot_w) - 2
            y1, y2 = 2, h - 2
            if x2 <= x1 or y2 <= y1:
                continue
            slot = buffer_img[y1:y2, x1:x2]

            # 判断该格子是否为空（暗色=空）
            gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) < 30:
                continue  # 空格子

            # 分类
            t_type, score = self.matcher.classify(slot)
            if score > 0.3:
                buffer_types.append(t_type)
            else:
                buffer_types.append("?")

        return buffer_types

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
        best_scores = []  # debug
        for bbox in candidates:
            x, y, w, h = bbox
            # 提取方块图像（留微小边距去噪）
            # 提取完整方块区域（模板已是干净图标，不需裁剪边距）
            icon_region = board_img[y:y + h, x:x + w]

            if icon_region.size == 0:
                continue

            tile_type, score = self.matcher.classify(icon_region)
            best_scores.append(score)

            if score >= TILE_MATCH_THRESHOLD or self.matcher.type_names == []:
                if not self.matcher.type_names:
                    tile_type = "unknown"
                tile = Tile(self.tile_counter, tile_type, bbox)
                tile.match_score = score
                tiles.append(tile)
                self.tile_counter += 1

        if best_scores:
            top5 = sorted(best_scores, reverse=True)[:5]
            print(f"[detector] 最高匹配分: {[f'{s:.3f}' for s in top5]} (阈值={TILE_MATCH_THRESHOLD})")

        print(f"[detector] 成功分类 {len(tiles)} 个方块")

        # 步骤3：亮度法判断可点击性
        self._compute_layers(tiles, board_img)

        return tiles

    def _find_tile_candidates(self, board_img: np.ndarray) -> list[tuple]:
        """
        混合检测：模板匹配 + 背景扣除，两路并进不漏方块
        """
        # 方法1: 全图滑窗模板匹配
        candidates = self._match_templates_full(board_img)

        # 方法2: 背景扣除（捕获模板匹配漏掉的方块）
        bg_candidates = self._find_by_background(board_img)

        # 合并
        all_candidates = candidates + bg_candidates
        if not all_candidates:
            return []

        # 去重
        merged = self._nms(all_candidates, iou_threshold=0.3)
        print(f"[detector] 混合检测: {len(merged)} 个候选 "
              f"(模板:{len(candidates)} 背景:{len(bg_candidates)})")
        return merged

    def _match_templates_full(self, board_img: np.ndarray) -> list[tuple]:
        """全图滑窗模板匹配"""
        candidates = []
        h, w = board_img.shape[:2]
        tw, th = TILE_SIZE
        scales = [0.80, 0.90, 1.0, 1.10]

        for t_type, templates in self.matcher.templates.items():
            for tmpl in templates:
                for scale in scales:
                    sw, sh = int(tw * scale), int(th * scale)
                    if sw > w or sh > h or sw < 10 or sh < 10:
                        continue
                    scaled_tmpl = cv2.resize(tmpl, (sw, sh))
                    try:
                        result = cv2.matchTemplate(board_img, scaled_tmpl,
                                                    cv2.TM_CCOEFF_NORMED)
                    except cv2.error:
                        continue

                    locations = np.where(result >= 0.45)
                    for pt in zip(*locations[::-1]):
                        candidates.append((pt[0], pt[1], sw, sh))

        return self._nms(candidates, iou_threshold=0.25)

    def _find_by_background(self, board_img: np.ndarray) -> list[tuple]:
        """回退：背景扣除法"""
        hsv = cv2.cvtColor(board_img, cv2.COLOR_BGR2HSV)

        # 尝试多个绿色范围，适配不同光照/纹理
        green_ranges = [
            ((25, 25, 30), (100, 255, 255)),   # 宽青绿
            ((30, 30, 40), (90, 255, 240)),     # 标准绿
            ((35, 20, 20), (85, 255, 255)),     # 深绿
        ]

        bg_mask = None
        for low, high in green_ranges:
            mask = cv2.inRange(hsv, low, high)
            if cv2.countNonZero(mask) > board_img.size * 0.05:
                bg_mask = mask
                break

        if bg_mask is None:
            return []

        # 形态学闭运算：填补绿色区域小孔
        kernel = np.ones((5, 5), np.uint8)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)

        # 取反：非绿色区域 = 方块
        tile_mask = cv2.bitwise_not(bg_mask)

        # 开运算去噪
        tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_OPEN,
                                      np.ones((3, 3), np.uint8))

        # 查找轮廓
        contours, _ = cv2.findContours(tile_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        img_area = board_img.shape[0] * board_img.shape[1]
        min_area = img_area * 0.00008   # 更小，捕获被遮挡方块
        max_area = img_area * 0.08

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if min_area <= area <= max_area:
                aspect = w / h if h > 0 else 0
                if 0.4 <= aspect <= 2.5:
                    candidates.append((x, y, w, h))

        candidates = self._merge_overlapping(candidates)
        return candidates

    def _nms(self, boxes: list[tuple], iou_threshold: float = 0.3) -> list[tuple]:
        """非极大值抑制，去重重叠框"""
        if len(boxes) <= 1:
            return boxes
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep = []
        for box in boxes:
            ok = True
            for kept in keep:
                if self._compute_iou(box, kept) > iou_threshold:
                    ok = False
                    break
            if ok:
                keep.append(box)
        return keep

    def _find_by_background(self, board_img: np.ndarray) -> list[tuple]:
        """回退：背景扣除法"""
        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 80)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        img_area = board_img.shape[0] * board_img.shape[1]
        min_area = img_area * 0.0002
        max_area = img_area * 0.06

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if min_area <= area <= max_area:
                aspect = w / h if h > 0 else 0
                if 0.4 <= aspect <= 2.5:
                    candidates.append((x, y, w, h))

        return self._merge_overlapping(candidates)

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

    def _compute_layers(self, tiles: list[Tile], board_img=None):
        """
        亮度法判断可点击：上层亮、下层暗。Otsu 自动阈值分割。
        """
        if not tiles:
            return

        if board_img is None:
            for t in tiles:
                t.free = True; t.layer = 0
            return

        gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
        brightnesses = []

        for tile in tiles:
            x, y, w, h = tile.bbox
            margin_x, margin_y = int(w * 0.2), int(h * 0.2)
            cx1 = max(0, x + margin_x)
            cy1 = max(0, y + margin_y)
            cx2 = min(gray.shape[1], x + w - margin_x)
            cy2 = min(gray.shape[0], y + h - margin_y)
            if cx2 > cx1 and cy2 > cy1:
                b = float(np.mean(gray[cy1:cy2, cx1:cx2]))
            else:
                b = 128.0
            brightnesses.append(b)

        if len(brightnesses) < 2:
            for t in tiles:
                t.free = True
            return

        arr = np.array(brightnesses, dtype=np.uint8)
        thresh, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for tile, b in zip(tiles, brightnesses):
            tile.free = b > thresh

        free_n = sum(1 for t in tiles if t.free)
        blocked_n = len(tiles) - free_n
        print(f"[detector] 亮度Otsu阈值={thresh} 可点击={free_n} 被挡={blocked_n} "
              f"亮度=[{min(brightnesses):.0f},{max(brightnesses):.0f}]")


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
