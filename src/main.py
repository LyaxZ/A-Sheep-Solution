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
from config import WECHAT_WINDOW_TITLE

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

        if not self.executor.calibrate():
            return False

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
        self.solver.revive_used = False

        for loop in range(200):
            if not running:
                return False

            print(f"\n-- Step {loop + 1} --")

            frame = get_game_screenshot()
            if frame is None:
                return False
            board = crop_game_board(frame)
            buf_img = crop_buffer(frame)

            # 检测棋盘
            tiles = self.detector.detect(board)
            if not tiles:
                print("No tiles detected")
                return False

            # 检测缓冲槽
            buffer_types = self.detector.detect_buffer(buf_img)
            from collections import Counter as _Ct
            buf_counts = _Ct(buffer_types)
            can_complete = [t for t, c in buf_counts.items() if c >= 2]
            buf_disp = " ".join(buffer_types) if buffer_types else "(empty)"
            if can_complete:
                buf_disp += f" [need 1 more: {can_complete}]"

            free_n = sum(1 for t in tiles if t.free)
            print(f"Board: {len(tiles)} tiles, {free_n} free | Buffer[{len(buffer_types)}]: {buf_disp}")

            # 求解
            result = self.solver.solve(tiles, buffer_types)

            if result.success:
                print(f"* Solution: {len(result.moves)} moves")
                executed = 0
                for tile_id, tile_type in result.moves[:8]:
                    if not running: break
                    for t in tiles:
                        if t.tile_type == tile_type and t.free:
                            self.executor.click_tile(t.bbox)
                            self.stats["moves"] += 1
                            executed += 1
                            t.free = False
                            time.sleep(0.12)
                            break
                if executed > 0:
                    print(f"  -> clicked {executed}")
                time.sleep(0.3)
                continue

            print(f"! {result.message}")
            buffer_full = len(buffer_types) >= 7
            item_key = user_item_prompt(
                result.recommended_item,
                revive_used=revive_used,
                buffer_full=buffer_full,
            )

            if item_key == "quit":
                running = False; return False
            if item_key is None:
                free_tiles = get_free_tiles(tiles)
                if free_tiles:
                    best = self._best_greedy_move(free_tiles, buffer_types)
                    if best:
                        self.executor.click_tile(best.bbox)
                        self.stats["moves"] += 1
                continue

            if item_key == "revive":
                revive_used = True
                self.solver.revive_used = True
            self.executor.click_item(item_key)
            self.stats["items_used"] += 1
            time.sleep(1.5)

        return False

    def _best_greedy_move(self, free_tiles, buffer_types: list):
        """贪心：优先三连 > 缓冲已有 > 释放上层"""
        from collections import Counter
        counts = Counter(buffer_types)
        for t in free_tiles:
            if counts.get(t.tile_type, 0) >= 2:
                return t
        for t in free_tiles:
            if counts.get(t.tile_type, 0) >= 1:
                return t
        return max(free_tiles, key=lambda t: t.match_score) if free_tiles else None

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
