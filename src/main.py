"""
羊了个羊 自动通关 - 主入口
用法:
  python src/main.py             正常运行（自动校准+开跑）
  python src/main.py --debug     调试：截图+检测方块，不点击
"""

import time
import signal
import sys

from capture import (
    find_wechat_window,
    get_game_screenshot,
    get_window_info,
    crop_game_board,
    save_debug_screenshot,
    auto_calibrate,
)
from detector import TileDetector, get_free_tiles
from solver import SheepSolver, format_solution
from executor import ClickExecutor, user_item_prompt
from config import WECHAT_WINDOW_TITLE

running = True


def signal_handler(sig, frame):
    global running
    print("\n⏹ 中断信号，退出...")
    running = False

signal.signal(signal.SIGINT, signal_handler)


class SheepAutoPlayer:
    """羊了个羊自动通关"""

    def __init__(self):
        self.detector = TileDetector()
        self.solver = SheepSolver(max_depth=60, max_backtrack=500)
        self.executor = ClickExecutor()
        self.stats = {"rounds": 0, "moves": 0, "items_used": 0}

    def setup(self) -> bool:
        """初始化 + 自动校准"""
        global running

        print("=" * 55)
        print("🐑 羊了个羊 自动通关")
        print("=" * 55)

        # 查找窗口
        win = find_wechat_window()
        if win is None:
            print(f"\n❌ 未找到小程序窗口 [{WECHAT_WINDOW_TITLE}]")
            print("请确保小程序已打开并进入游戏关卡")
            return False

        print(f"✅ 窗口: {win.width}x{win.height}")

        # 自动校准区域
        if not auto_calibrate():
            print("⚠ 自动校准失败，使用默认比例")

        # 初始化执行器
        if not self.executor.calibrate():
            return False

        # 检查模板
        type_count = len(self.detector.matcher.type_names)
        if type_count == 0:
            print("\n⚠ 未找到方块模板！")
            print(f"  请将方块图标截图放到: data/templates/")
            print(f"  命名随意 (如 1.png 2.png ...)")
            print(f"  每种图标截一张即可，约 10-15 张")
            return False

        print(f"📦 模板: {type_count} 种")

        if not running:
            return False

        print("\n" + "=" * 55)
        print("3 秒后自动开始，Ctrl+C 停止")
        print("=" * 55)
        time.sleep(3)
        return True

    def run(self):
        global running
        if not self.setup():
            return

        while running:
            self.stats["rounds"] += 1
            print(f"\n{'─'*40}")
            print(f"🎮 第 {self.stats['rounds']} 局")
            print(f"{'─'*40}")
            time.sleep(1.0)

            success = self._play_one_round()
            if success:
                print("\n🎉 通关！")
                break

            print("\n💀 本局失败")
            if not running:
                break

            c = input("\n重试? (y/n): ").strip().lower()
            if c != "y":
                break

            print("重新开始...")
            self.executor.click_restart()
            time.sleep(2.0)

        self._print_stats()

    def _play_one_round(self) -> bool:
        global running
        max_loops = 200
        revive_used = False
        self.solver.revive_used = False

        for loop in range(max_loops):
            if not running:
                return False

            print(f"\n--- 轮次 {loop + 1} ---")

            # 截图
            win_info = get_window_info()
            frame = get_game_screenshot(win_info)
            if frame is None:
                return False
            board = crop_game_board(frame)

            # 识别
            tiles = self.detector.detect(board)
            if not tiles:
                print("未检测到方块")
                return False

            free_n = sum(1 for t in tiles if t.free)
            print(f"棋盘: {len(tiles)} 方块, {free_n} 可点击")

            # 求解
            result = self.solver.solve(tiles)

            if result.success:
                print(f"✅ 路径 {len(result.moves)} 步")
                self._execute_moves(result.moves)
                continue

            # 求解失败 → 道具
            print(f"❌ {result.message}")
            buffer_full = "死胡同" in result.message or "缓冲" in result.message

            item_key = user_item_prompt(
                result.recommended_item,
                revive_used=revive_used,
                buffer_full=buffer_full,
            )

            if item_key == "quit":
                running = False
                return False

            if item_key is None:
                # 跳过道具，贪心走一步
                free_tiles = get_free_tiles(tiles)
                if free_tiles:
                    best = max(free_tiles, key=lambda t: t.layer)
                    self.executor.click_tile(best.bbox)
                    self.stats["moves"] += 1
                continue

            # 使用道具
            if item_key == "revive":
                revive_used = True
                self.solver.revive_used = True
                print("💀 复活 (槽满移除左侧3个)")

            self.executor.click_item(item_key)
            self.stats["items_used"] += 1
            time.sleep(1.5)

        print(f"\n⚠ 超过最大循环 ({max_loops})")
        return False

    def _execute_moves(self, moves: list):
        for i, (tile_id, tile_type) in enumerate(moves):
            if not running:
                return
            self.stats["moves"] += 1
        time.sleep(0.5)

    def _print_stats(self):
        print(f"\n{'='*40}")
        print(f"📊 总局: {self.stats['rounds']}  点击: {self.stats['moves']}  道具: {self.stats['items_used']}")
        print(f"{'='*40}")


# ---- 主入口 ----

def main():
    import argparse
    parser = argparse.ArgumentParser(description="羊了个羊自动通关")
    parser.add_argument("--debug", action="store_true", help="截图+检测方块，不点击")
    args = parser.parse_args()

    if args.debug:
        from capture import find_wechat_window, get_game_screenshot, crop_game_board, save_debug_screenshot
        win = find_wechat_window()
        if not win:
            print(f"❌ 未找到窗口 [{WECHAT_WINDOW_TITLE}]")
            return
        frame = get_game_screenshot()
        if frame is not None:
            save_debug_screenshot(frame, "debug_full")
            board = crop_game_board(frame)
            save_debug_screenshot(board, "debug_board")
            detector = TileDetector()
            tiles = detector.detect(board)
            print(f"\n检测: {len(tiles)} 方块")
            for t in tiles[:25]:
                print(f"  {t}")
    else:
        player = SheepAutoPlayer()
        player.run()


if __name__ == "__main__":
    main()
