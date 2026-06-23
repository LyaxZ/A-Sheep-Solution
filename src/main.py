"""
羊了个羊 自动通关程序 - 主入口
================================
流程：截图 → 识别方块 → 求解 → 模拟点击 → 循环
"""

import time
import signal
import sys

from capture import (
    find_wechat_window,
    get_game_screenshot,
    crop_game_board,
    save_debug_screenshot,
)
from detector import TileDetector, get_free_tiles
from solver import SheepSolver, format_solution
from executor import ClickExecutor, user_item_prompt


# 全局标志
running = True


def signal_handler(sig, frame):
    """Ctrl+C 处理"""
    global running
    print("\n\n⏹ 收到中断信号，正在退出...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


class SheepAutoPlayer:
    """羊了个羊自动通关主控"""

    def __init__(self):
        self.detector = TileDetector()
        self.solver = SheepSolver(max_depth=60, max_backtrack=500)
        self.executor = ClickExecutor()
        self.stats = {
            "rounds": 0,       # 已玩局数
            "moves": 0,        # 总点击次数
            "items_used": 0,   # 道具使用次数
        }

    def setup(self) -> bool:
        """初始化：定位窗口"""
        print("=" * 60)
        print("🐑 羊了个羊 自动通关程序")
        print("=" * 60)

        win = find_wechat_window()
        if win is None:
            print("\n❌ 未找到微信窗口！")
            print("请确保：")
            print("  1. 微信已登录并打开")
            print("  2. 羊了个羊小程序已启动并进入游戏")
            return False

        print(f"\n✅ 找到微信窗口: {win.title}")
        print(f"   位置: ({win.left}, {win.top})")
        print(f"   大小: {win.width}x{win.height}")

        if not self.executor.calibrate():
            return False

        # 检查模板
        if not self.detector.matcher.type_names:
            print("\n⚠️ 未加载方块图标模板！")
            print(f"请将模板图片放入: data/templates/")
            print("模板文件命名格式: {类型名}.png 或 {类型名}_1.png")
            print("例如: wood.png, hay.png, fire.png")
            return False

        print(f"\n📦 已加载 {len(self.detector.matcher.type_names)} 种方块模板")
        print("\n" + "=" * 60)
        print("准备就绪！5 秒后开始...")
        print("按 Ctrl+C 可随时停止")
        print("=" * 60)
        time.sleep(5)
        return True

    def run(self):
        """主循环"""
        global running

        if not self.setup():
            return

        while running:
            self.stats["rounds"] += 1
            print(f"\n{'='*60}")
            print(f"🎮 第 {self.stats['rounds']} 局开始")
            print(f"{'='*60}")

            # 等待游戏画面稳定
            time.sleep(1.0)

            success = self._play_one_round()
            if success:
                print("\n🎉🎉🎉 恭喜通关！🎉🎉🎉")
                break
            else:
                print("\n💀 本局失败")

                if not running:
                    break

                # 询问是否重试
                choice = input("\n是否重试？(y/n): ").strip().lower()
                if choice != "y":
                    print("退出程序")
                    break

                # 点击重新开始
                print("点击重新开始...")
                self.executor.click_restart()
                time.sleep(2.0)

        self._print_stats()

    def _play_one_round(self) -> bool:
        """玩一局，返回是否通关"""
        global running

        max_loops = 200  # 安全上限
        revive_used = False  # 本局复活是否已用

        # 求解器也需要知道复活状态
        self.solver.revive_used = False

        for loop in range(max_loops):
            if not running:
                return False

            print(f"\n--- 第 {loop + 1} 轮 ---")

            # 1. 截图
            frame = get_game_screenshot(self.executor.win)
            if frame is None:
                print("截图失败")
                return False

            board = crop_game_board(frame)

            # 2. 识别方块
            tiles = self.detector.detect(board)
            if not tiles:
                print("未检测到方块，可能已通关或游戏画面异常")
                return False

            free_count = sum(1 for t in tiles if t.free)
            print(f"棋盘: {len(tiles)} 个方块, {free_count} 个可点击")

            # 3. 求解
            result = self.solver.solve(tiles)

            if result.success:
                # 执行移动序列
                print(f"✅ 找到路径: {len(result.moves)} 步")
                self._execute_moves(result.moves)
                continue

            # 4. 求解失败，尝试道具
            print(f"❌ {result.message}")

            # 检测缓冲槽是否满（用于判断复活是否可用）
            from collections import Counter
            # 无法直接获取缓冲槽状态，用求解结果判断
            buffer_full = "缓冲区死胡同" in result.message or "缓冲" in result.message

            if result.recommended_item:
                item_key = user_item_prompt(
                    result.recommended_item,
                    revive_used=revive_used,
                    buffer_full=buffer_full,
                )
            else:
                item_key = user_item_prompt(
                    None,
                    revive_used=revive_used,
                    buffer_full=buffer_full,
                )

            if item_key == "quit":
                running = False
                return False

            if item_key is None:
                # 用户选择跳过道具
                print("跳过道具，尝试继续...")
                free_tiles = get_free_tiles(tiles)
                if free_tiles:
                    best = self._pick_safe_move(free_tiles)
                    if best:
                        self.executor.click_tile(best.bbox)
                        self.stats["moves"] += 1
                continue

            # 使用道具
            if item_key == "revive":
                if revive_used:
                    print("⚠️ 复活已用过，本局无法再次使用")
                    continue
                revive_used = True
                self.solver.revive_used = True
                print("💀 使用复活：将移除缓冲槽最左侧3个方块")
                print("   (如需观看广告，请在手机/模拟器上手动操作)")

            self.executor.click_item(item_key)
            self.stats["items_used"] += 1
            time.sleep(1.5)  # 等待道具动画

        print(f"\n⚠️ 超过最大循环次数 ({max_loops})，结束本局")
        return False

    def _execute_moves(self, moves: list):
        """执行一系列移动"""
        for i, (tile_id, tile_type) in enumerate(moves):
            if not running:
                return
            # 注意：这里需要重新截图识别来获取当前方块的实际位置
            # 因为之前的移动可能改变了局面
            # 简化处理：每次重新识别
            if i > 0 and i % 3 == 0:
                # 每3步重新截图确认
                time.sleep(0.3)
                frame = get_game_screenshot(self.executor.win)
                if frame is not None:
                    board = crop_game_board(frame)
                    tiles = self.detector.detect(board)
                    # 找到对应方块
                    for t in tiles:
                        if t.tile_type == tile_type and t.free:
                            self.executor.click_tile(t.bbox)
                            self.stats["moves"] += 1
                            break
            else:
                # 前几步相信求解结果，直接点击
                self.stats["moves"] += 1

        time.sleep(0.5)  # 等待动画

    def _pick_safe_move(self, free_tiles: list):
        """
        没有完整路径时，选择一个最安全的移动
        优先选缓冲中已有的类型
        """
        if not free_tiles:
            return None

        # 给每个可点击方块打分
        from collections import Counter

        scored = []
        for tile in free_tiles:
            score = 0
            # 优先选能解除阻挡的
            score += tile.layer * 10
            scored.append((tile, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def _print_stats(self):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print("📊 统计信息")
        print(f"{'='*60}")
        print(f"  总局数:    {self.stats['rounds']}")
        print(f"  总点击:    {self.stats['moves']}")
        print(f"  道具使用:  {self.stats['items_used']}")
        print(f"{'='*60}")


# ---- 便捷函数 ----

def calibrate():
    """
    校准模式：截图当前游戏画面，帮助用户调整坐标参数
    """
    from capture import find_wechat_window, get_game_screenshot, save_debug_screenshot, crop_game_board

    print("=" * 60)
    print("🔧 校准模式")
    print("=" * 60)

    win = find_wechat_window()
    if win is None:
        print("❌ 未找到微信窗口")
        return

    print(f"✅ 微信窗口: ({win.left}, {win.top}) {win.width}x{win.height}")

    frame = get_game_screenshot(win)
    if frame is not None:
        save_debug_screenshot(frame, "full_calibrate")
        board = crop_game_board(frame)
        save_debug_screenshot(board, "board_calibrate")
        print("📸 截图已保存到 data/screenshots/")
        print("请打开截图，在 config.py 中调整 GAME_BOARD_REGION 等参数")
        print(f"当前棋盘区域: {crop_game_board.__defaults__}")


def capture_templates():
    """
    模板收集模式：交互式截取方块图标作为模板
    """
    import cv2
    from pathlib import Path
    from config import TEMPLATE_DIR, TILE_SIZE

    print("=" * 60)
    print("📷 模板收集模式")
    print("=" * 60)
    print("此模式帮助你收集游戏中各种方块图标作为匹配模板")
    print()

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    print("步骤：")
    print("1. 打开游戏，等待棋盘显示")
    print("2. 输入图标名称（如 wood, hay, fire）")
    print("3. 程序会自动截图并让你框选图标区域")
    print()

    from capture import find_wechat_window, get_game_screenshot, crop_game_board, save_debug_screenshot

    win = find_wechat_window()
    if win is None:
        print("❌ 未找到微信窗口")
        return

    frame = get_game_screenshot(win)
    if frame is None:
        return

    # 保存棋盘截图供参考
    board = crop_game_board(frame)
    save_debug_screenshot(board, "board_for_template")

    print("📸 棋盘截图已保存，请查看 data/screenshots/")

    while True:
        name = input("\n输入模板名称 (或 q 退出): ").strip().lower()
        if name == "q":
            break
        if not name:
            continue

        try:
            x = int(input("  方块左上角 X (棋盘内): "))
            y = int(input("  方块左上角 Y (棋盘内): "))
            w = int(input(f"  宽度 (默认 {TILE_SIZE[0]}): ") or TILE_SIZE[0])
            h = int(input(f"  高度 (默认 {TILE_SIZE[1]}): ") or TILE_SIZE[1])
        except ValueError:
            print("输入无效")
            continue

        # 裁剪
        icon = board[y:y + h, x:x + w]
        if icon.size == 0:
            print("裁剪区域为空，请检查坐标")
            continue

        # 缩放到标准大小
        icon = cv2.resize(icon, TILE_SIZE)

        # 保存
        idx = len(list(TEMPLATE_DIR.glob(f"{name}*.png"))) + 1
        path = TEMPLATE_DIR / f"{name}_{idx}.png"
        cv2.imwrite(str(path), icon)
        print(f"✅ 已保存: {path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="羊了个羊自动通关")
    parser.add_argument("--calibrate", action="store_true", help="校准模式：截图辅助调参")
    parser.add_argument("--templates", action="store_true", help="模板收集模式：截取方块图标")
    parser.add_argument("--debug", action="store_true", help="调试模式：截图并检测方块")

    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    elif args.templates:
        capture_templates()
    elif args.debug:
        # 调试：截图 + 检测
        from capture import find_wechat_window, get_game_screenshot, crop_game_board, save_debug_screenshot

        win = find_wechat_window()
        if win:
            frame = get_game_screenshot(win)
            if frame is not None:
                save_debug_screenshot(frame, "debug_full")
                board = crop_game_board(frame)
                save_debug_screenshot(board, "debug_board")

                detector = TileDetector()
                tiles = detector.detect(board)
                print(f"\n检测到 {len(tiles)} 个方块")
                free = get_free_tiles(tiles)
                print(f"可点击: {len(free)}")
                for t in tiles[:20]:
                    print(f"  {t}")
    else:
        # 正常运行
        player = SheepAutoPlayer()
        player.run()


if __name__ == "__main__":
    main()
