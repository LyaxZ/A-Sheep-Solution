"""
羊了个羊 自动通关 - 主入口
用法:
  python src/main.py             正常运行
  python src/main.py --debug     调试：截图+检测，不点击
"""

import time
import signal
import sys
import argparse

from capture import (
    find_wechat_window,
    get_game_screenshot,
    get_window_info,
    crop_game_board,
    crop_buffer,
    save_debug_screenshot,
    auto_calibrate,
)
from detector import TileDetector, get_free_tiles
from solver import SheepSolver
from executor import ClickExecutor, user_item_prompt
from config import WECHAT_WINDOW_TITLE, GAME_BOARD_RATIO

running = True


class SheepAutoPlayer:

    def __init__(self):
        self.detector = TileDetector()
        self.solver = SheepSolver(max_depth=60, max_backtrack=500)
        self.executor = ClickExecutor()
        self.stats = {"rounds": 0, "moves": 0, "items_used": 0}

    def setup(self) -> bool:
        print("=" * 50)
        print("Sheep Auto Solver")
        print("=" * 50)

        win = find_wechat_window()
        if win is None:
            print(f"[!] Window not found: {WECHAT_WINDOW_TITLE}")
            print("    Make sure the mini-program is open and in a game level")
            return False

        print(f"[*] Window: {win.width}x{win.height}")

        if not auto_calibrate():
            print("[!] Auto-calibrate failed, using default ratios")

        # 用截图修正DPI
        winfo = get_window_info()
        get_game_screenshot(winfo)  # 修正DPI
        if not self.executor.calibrate():
            return False
        self.executor.win_info = winfo  # 同步修正后尺寸

        type_count = len(self.detector.matcher.type_names)
        if type_count == 0:
            print("[!] No tile templates found!")
            print(f"    Put tile icon PNGs into: data/templates/")
            print(f"    Name them anything (e.g. 1.png 2.png ...)")
            return False

        print(f"[*] Templates: {type_count} types")
        print("\nStarting in 3 seconds... Ctrl+C to stop")
        time.sleep(3)
        return True

    def run(self):
        global running
        if not self.setup():
            return

        while running:
            self.stats["rounds"] += 1
            print(f"\n{'='*40}")
            print(f"Round {self.stats['rounds']}")
            print(f"{'='*40}")
            time.sleep(1.0)

            if self._play_one_round():
                print("\n*** CLEARED! ***")
                break

            print("\nRound failed")
            if not running:
                break

            c = input("\nRetry? (y/n): ").strip().lower()
            if c != "y":
                break
            print("Restarting...")
            self.executor.click_restart()
            time.sleep(2.0)

        self._print_stats()

    def _play_one_round(self) -> bool:
        global running
        revive_used = False
        stuck_count = 0
        last_state = ""
        last_click_pos = ""

        for step in range(300):
            if not running:
                return False

            winfo = get_window_info()
            frame = get_game_screenshot(winfo)
            if frame is None: return False
            self.executor.win_info = winfo
            board = crop_game_board(frame)
            buf_img = crop_buffer(frame)

            tiles = self.detector.detect(board)
            if not tiles:
                print("Board empty - CLEARED!")
                return True

            buffer_types = self.detector.detect_buffer(buf_img)

            # 缓冲全同一类型 = 检测错误，忽略缓冲
            if len(buffer_types) >= 4 and len(set(buffer_types)) == 1:
                buffer_types = []

            free_tiles = [t for t in tiles if t.free]
            blocked_tiles = [t for t in tiles if not t.free]

            from collections import Counter
            buf_counts = Counter(buffer_types)
            free_counts = Counter(t.tile_type for t in free_tiles)

            # 死循环检测（含位置）
            ft = tuple(sorted((t.tile_type, t.bbox[0], t.bbox[1]) for t in free_tiles))
            state_key = f"{ft}|{tuple(sorted(buffer_types))}"
            if state_key == last_state:
                stuck_count += 1
            else:
                stuck_count = 0
            last_state = state_key

            if stuck_count >= 2:
                print(f"  !! Stuck, try other")
                if len(free_tiles) > 1:
                    self._click_tile(free_tiles[-1])
                elif free_tiles:
                    self._click_tile(free_tiles[0])
                last_click_pos = ""; stuck_count = 0
                continue

            buf_disp = " ".join(buffer_types) if buffer_types else "(empty)"
            print(f"\n-- Step {step+1} --")
            print(f"Board: {len(tiles)}({len(free_tiles)}f/{len(blocked_tiles)}b) "
                  f"Buf[{len(buffer_types)}]: {buf_disp}")

            acted = False

            # 策略1: 缓冲有≥2个 → 完成三连（最高优先级）
            for t_type, cnt in buf_counts.most_common():
                if cnt >= 2:
                    for t in free_tiles:
                        if t.tile_type == t_type:
                            pk = f"{t.bbox[0]},{t.bbox[1]}"
                            if pk == last_click_pos: continue
                            print(f"  -> Complete triple [{t_type}]! (buf={cnt})")
                            last_click_pos = pk
                            self._click_tile(t); acted = True; break
                if acted: break

            # 策略2: 缓冲有1个 → 凑对（第二优先级）
            if not acted:
                for t_type, cnt in buf_counts.most_common():
                    if cnt >= 1:
                        for t in free_tiles:
                            if t.tile_type == t_type:
                                pk = f"{t.bbox[0]},{t.bbox[1]}"
                                if pk == last_click_pos: continue
                                print(f"  -> Match buffer [{t_type}] (buf={cnt})")
                                last_click_pos = pk
                                self._click_tile(t); acted = True; break
                    if acted: break

            # 策略3: 场上有3同类型 → 全点
            if not acted:
                for t_type, cnt in free_counts.most_common():
                    if cnt >= 3:
                        matches = [t for t in free_tiles if t.tile_type == t_type][:3]
                        print(f"  -> Triple [{t_type}] x3")
                        for t in matches:
                            self._click_tile(t)
                        acted = True; break

            # 策略4: 解锁
            if not acted:
                best = self._find_best_unlock(free_tiles, blocked_tiles, buf_counts)
                if best:
                    print(f"  -> Unlock [{best[0]}]")
                    self._click_tile(best[1]); acted = True

            # 策略5: 释放最多被挡方块
            if not acted and free_tiles:
                best = max(free_tiles, key=lambda t: self._count_blocked_by(t, blocked_tiles))
                print(f"  -> Click [{best.tile_type}]")
                self._click_tile(best); acted = True

            # 策略6: 道具
            if not acted:
                print("! No moves")
                item_key = user_item_prompt(
                    None, revive_used=revive_used,
                    buffer_full=len(buffer_types) >= 7,
                )
                if item_key == "quit":
                    running = False; return False
                if item_key is None: break
                if item_key == "revive": revive_used = True
                self.executor.click_item(item_key)
                self.stats["items_used"] += 1
                time.sleep(1.5)

        return False

    def _click_tile(self, tile):
        """点击方块并更新统计"""
        x, y, w, h = tile.bbox
        bx, by, _, _ = self.executor.win_info.ratio_to_region(GAME_BOARD_RATIO)
        sx = self.executor.win_info.client_left + bx + x + w // 2
        sy = self.executor.win_info.client_top + by + y + h // 2
        print(f"  click [{tile.tile_type}] screen=({sx},{sy}) win={self.executor.win_info.width}x{self.executor.win_info.height}")
        self.executor.click_tile(tile.bbox)
        self.stats["moves"] += 1
        time.sleep(0.15)

    def _find_best_unlock(self, free_tiles, blocked_tiles, buf_counts):
        """
        找到最优解锁：点击哪个可点击方块，能释放最多缓冲中已有类型的被挡方块。
        返回 (target_type, tile_to_click) 或 None
        """
        best_score = -1
        best = None

        for free_tile in free_tiles:
            # 计算点击此方块后，有多少缓冲类型的被挡方块会被释放
            score = 0
            target_type = None
            for blocked in blocked_tiles:
                # 检查 free_tile 是否遮挡了 blocked
                if self._tile_blocks(free_tile, blocked):
                    bt = blocked.tile_type
                    # 缓冲中有此类型 → 高分
                    buf_cnt = buf_counts.get(bt, 0)
                    if buf_cnt >= 2:
                        score += 100  # 释放后能直接三连
                    elif buf_cnt >= 1:
                        score += 30
                    elif buf_cnt == 0:
                        score += 5  # 没有缓冲也有价值
                    if score > best_score:
                        target_type = bt

            if score > best_score:
                best_score = score
                best = (target_type, free_tile)

        return best if best_score > 0 else None

    def _tile_blocks(self, upper, lower) -> bool:
        """upper 是否遮挡了 lower"""
        ax, ay, aw, ah = upper.bbox
        bx, by, bw, bh = lower.bbox
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2 - ix1) * (iy2 - iy1)
            return inter > lower.area * 0.05
        return False

    def _count_blocked_by(self, tile, blocked_tiles) -> int:
        """计算 tile 遮挡了多少被挡方块"""
        return sum(1 for b in blocked_tiles if self._tile_blocks(tile, b))

    def _print_stats(self):
        print(f"\n{'='*40}")
        print(f"Rounds: {self.stats['rounds']}  Clicks: {self.stats['moves']}  Items: {self.stats['items_used']}")
        print(f"{'='*40}")


def main():
    global running

    def handler(sig, frame):
        global running
        print("\nInterrupted, exiting...")
        running = False

    signal.signal(signal.SIGINT, handler)

    parser = argparse.ArgumentParser(description="Sheep Auto Solver")
    parser.add_argument("--debug", action="store_true", help="Screenshot + detect only")
    args = parser.parse_args()

    if args.debug:
        win = find_wechat_window()
        if not win:
            print(f"[!] Window not found: {WECHAT_WINDOW_TITLE}")
            return
        frame = get_game_screenshot()
        if frame is not None:
            save_debug_screenshot(frame, "debug_full")
            board = crop_game_board(frame)
            save_debug_screenshot(board, "debug_board")
            detector = TileDetector()
            tiles = detector.detect(board)
            print(f"\nDetected: {len(tiles)} tiles")
            for t in tiles[:30]:
                print(f"  {t}")
    else:
        player = SheepAutoPlayer()
        player.run()


if __name__ == "__main__":
    main()
