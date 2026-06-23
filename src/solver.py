"""
求解算法模块 - 基于 DFS + 启发式搜索找到通关路径
"""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field

from config import BUFFER_MAX, MATCH_COUNT, ITEMS


@dataclass
class GameState:
    """游戏状态快照"""
    tiles: list          # 所有剩余方块 (Tile 对象)
    buffer: list         # 缓冲槽中的类型列表
    move_history: list   # 已执行的移动 [(tile_id, tile_type), ...]

    def clone(self):
        return GameState(
            tiles=deepcopy(self.tiles),
            buffer=list(self.buffer),
            move_history=list(self.move_history),
        )


@dataclass
class SolverResult:
    """求解结果"""
    success: bool
    moves: list = field(default_factory=list)        # 通关的点击序列
    recommended_item: str | None = None              # 推荐使用的道具 key
    message: str = ""


class SheepSolver:
    """
    羊了个羊求解器
    使用贪心 + 有限回溯 DFS 搜索可行路径
    """

    def __init__(self, max_depth: int = 50, max_backtrack: int = 200):
        self.max_depth = max_depth          # 最大搜索深度
        self.max_backtrack = max_backtrack  # 最大回溯步数
        self.visited_states: set = set()    # 已访问状态哈希（剪枝）
        self.nodes_explored = 0
        self.revive_used = False            # 本局是否已使用复活

    def solve(self, tiles: list) -> SolverResult:
        """
        主求解入口
        tiles: 当前棋盘上的所有方块列表
        """
        self.visited_states.clear()
        self.nodes_explored = 0

        # 检查方块总数是否为 3 的倍数
        total_tiles = len(tiles)
        if total_tiles % MATCH_COUNT != 0:
            return SolverResult(
                success=False,
                message=f"方块总数 {total_tiles} 不是 {MATCH_COUNT} 的倍数，可能检测有误"
            )

        # 统计每种类型的数量是否都是 3 的倍数
        type_counts = Counter(t.tile_type for t in tiles)
        for t_type, cnt in type_counts.items():
            if cnt % MATCH_COUNT != 0:
                return SolverResult(
                    success=False,
                    message=f"类型 [{t_type}] 数量 {cnt} 不是 {MATCH_COUNT} 的倍数，可能分类有误"
                )

        initial_state = GameState(tiles=list(tiles), buffer=[], move_history=[])

        print(f"[solver] 开始求解: {total_tiles} 个方块, "
              f"{sum(1 for t in tiles if t.free)} 个可点击, "
              f"{len(type_counts)} 种类型")

        # 先尝试贪心求解（快速路径）
        result = self._greedy_solve(initial_state)
        if result.success:
            return result

        print(f"[solver] 贪心未解，尝试回溯搜索...")

        # 贪心失败，尝试 DFS 回溯
        result = self._dfs_solve(initial_state, depth=0)
        if result.success:
            return result

        # 求解失败，推荐道具
        item_key = self._recommend_item(initial_state)
        if item_key:
            item = ITEMS.get(item_key)
            return SolverResult(
                success=False,
                recommended_item=item_key,
                message=f"自动求解失败，建议使用道具: {item.name if item else item_key}"
            )

        return SolverResult(
            success=False,
            message="自动求解失败，无法推荐道具，建议重开"
        )

    def _greedy_solve(self, state: GameState) -> SolverResult:
        """贪心求解：每步选最优方块"""
        max_steps = len(state.tiles) * 2  # 安全上限

        for step in range(max_steps):
            free_tiles = [t for t in state.tiles if t.free]
            if not free_tiles:
                break

            if not state.tiles:
                # 全部消除！
                return SolverResult(
                    success=True,
                    moves=list(state.move_history),
                    message=f"贪心求解成功，共 {len(state.move_history)} 步"
                )

            # 选择最优方块
            best_tile = self._select_best_tile(free_tiles, state.buffer, state.tiles)
            if best_tile is None:
                break

            # 执行移动
            self._apply_move(state, best_tile)

        # 贪心没解完，返回失败
        return SolverResult(success=False, message="贪心求解未完成")

    def _dfs_solve(self, state: GameState, depth: int) -> SolverResult:
        """DFS 回溯搜索"""
        self.nodes_explored += 1

        # 终止条件
        if not state.tiles:
            return SolverResult(
                success=True,
                moves=list(state.move_history),
                message=f"DFS求解成功，共 {len(state.move_history)} 步"
            )

        if depth >= self.max_depth:
            return SolverResult(success=False)

        if self.nodes_explored > self.max_backtrack:
            return SolverResult(success=False, message="超过回溯上限")

        # 状态哈希剪枝
        state_hash = self._hash_state(state)
        if state_hash in self.visited_states:
            return SolverResult(success=False)
        self.visited_states.add(state_hash)

        # 死胡同检测：缓冲区满且无三消
        if self._is_dead_end(state):
            return SolverResult(success=False, message="缓冲区死胡同")

        # 获取可点击方块
        free_tiles = [t for t in state.tiles if t.free]
        if not free_tiles:
            return SolverResult(success=False, message="无可点击方块")

        # 按启发式排序
        scored_tiles = self._score_tiles(free_tiles, state.buffer, state.tiles)
        scored_tiles.sort(key=lambda x: x[1], reverse=True)

        # 分支搜索
        for tile, _score in scored_tiles[:8]:  # 限制分支因子
            new_state = state.clone()
            self._apply_move(new_state, tile)

            result = self._dfs_solve(new_state, depth + 1)
            if result.success:
                return result

        return SolverResult(success=False)

    # ---- 核心逻辑 ----

    def _apply_move(self, state: GameState, tile):
        """执行一次点击：从棋盘移除方块，加入缓冲槽，检测消除"""
        # 从棋盘移除
        state.tiles = [t for t in state.tiles if t.id != tile.id]

        # 加入缓冲槽
        state.buffer.append(tile.tile_type)
        state.move_history.append((tile.id, tile.tile_type))

        # 检测三消
        self._check_match(state)

        # 更新可点击状态（被这个方块挡住的方块可能现在可点击了）
        self._update_free_status(state.tiles)

    def _check_match(self, state: GameState):
        """检查缓冲槽中是否有 3 个相同方块，有则消除"""
        counts = Counter(state.buffer)
        for t_type, cnt in counts.items():
            if cnt >= MATCH_COUNT:
                # 移除 3 个该类型
                removed = 0
                new_buffer = []
                for item in state.buffer:
                    if item == t_type and removed < MATCH_COUNT:
                        removed += 1
                    else:
                        new_buffer.append(item)
                state.buffer = new_buffer

    def _update_free_status(self, tiles: list):
        """重新计算所有方块的可点击状态"""
        if not tiles:
            return

        # 先全部设为可点击
        for t in tiles:
            t.free = True

        # 按层排序
        layers = sorted(set(t.layer for t in tiles), reverse=True)
        if not layers:
            return

        # 从上层到下检查遮挡
        for tile in tiles:
            cx, cy = tile.center
            for other in tiles:
                if other.id == tile.id:
                    continue
                ox, oy, ow, oh = other.bbox
                if (ox <= cx <= ox + ow) and (oy <= cy <= oy + oh):
                    if other.layer > tile.layer:
                        tile.free = False
                        break

    def _is_dead_end(self, state: GameState) -> bool:
        """
        判断是否死胡同：
        缓冲槽已满（≥BUFFER_MAX）且没有任何类型达到可消除数量
        """
        if len(state.buffer) < BUFFER_MAX:
            return False
        counts = Counter(state.buffer)
        return not any(c >= MATCH_COUNT for c in counts.values())

    # ---- 启发式选择 ----

    def _select_best_tile(self, free_tiles: list, buffer: list, all_tiles: list):
        """
        贪心选择最优方块：
        1. 优先选能立即完成三消的
        2. 其次选缓冲中已有同类型的
        3. 再选能释放最多方块的
        """
        scored = self._score_tiles(free_tiles, buffer, all_tiles)
        if not scored:
            return None
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def _score_tiles(self, free_tiles: list, buffer: list, all_tiles: list) -> list:
        """
        对可选方块打分，返回 [(tile, score), ...]
        分数越高越优先
        """
        buffer_counts = Counter(buffer)
        scored = []

        for tile in free_tiles:
            score = 0

            # 加分1：能立即完成三消（最大权重）
            if buffer_counts.get(tile.tile_type, 0) >= MATCH_COUNT - 1:
                score += 1000

            # 加分2：缓冲中已有同类型（接近消除）
            elif buffer_counts.get(tile.tile_type, 0) >= 1:
                score += 200

            # 加分3：点击后能释放多少被挡方块
            freed = self._count_freed_tiles(tile, all_tiles)
            score += freed * 50

            # 扣分1：缓冲槽快满时，选新类型有风险
            if len(buffer) >= BUFFER_MAX - 1:
                if buffer_counts.get(tile.tile_type, 0) == 0:
                    score -= 500

            # 扣分2：缓冲中已有2个不同类型，选第三种有风险
            unique_count = len(set(buffer))
            if unique_count >= 4 and buffer_counts.get(tile.tile_type, 0) == 0:
                score -= 100

            # 加分4：优先消耗数量少的类型（减少残留风险）
            type_total = sum(1 for t in all_tiles if t.tile_type == tile.tile_type)
            if type_total <= MATCH_COUNT:
                score += 300  # 只剩一组，优先消掉

            scored.append((tile, score))

        return scored

    def _count_freed_tiles(self, clicked_tile, all_tiles: list) -> int:
        """计算点击某个方块后，有多少被挡方块会被释放"""
        count = 0
        cx, cy = clicked_tile.center
        for tile in all_tiles:
            if tile.id == clicked_tile.id or tile.free:
                continue
            # 如果被点击方块是唯一遮挡该方块的，则该方块会被释放
            ox, oy, ow, oh = clicked_tile.bbox
            tx, ty = tile.center
            if (ox <= tx <= ox + ow) and (oy <= ty <= oy + oh):
                # 检查是否还有其他方块也遮挡它
                blocked_by_others = False
                for other in all_tiles:
                    if other.id in (clicked_tile.id, tile.id):
                        continue
                    bx, by, bw, bh = other.bbox
                    if (bx <= tx <= bx + bw) and (by <= ty <= by + bh):
                        if other.layer > tile.layer:
                            blocked_by_others = True
                            break
                if not blocked_by_others:
                    count += 1
        return count

    # ---- 状态哈希 ----

    def _hash_state(self, state: GameState) -> int:
        """计算状态哈希值用于去重"""
        free_ids = tuple(sorted(t.id for t in state.tiles if t.free))
        buffer_tuple = tuple(sorted(state.buffer))
        return hash((free_ids, buffer_tuple))

    # ---- 道具推荐 ----

    def _recommend_item(self, state: GameState) -> str | None:
        """分析局面，推荐最合适的道具（含复活）
        
        关键规则：槽满时洗牌/撤销/移出不可用，只有复活有效！
        """
        buffer = state.buffer
        free_tiles = [t for t in state.tiles if t.free]
        buffer_counts = Counter(buffer)
        unique_in_buffer = len(buffer_counts)
        buffer_full = len(buffer) >= BUFFER_MAX

        # 统计被阻塞的方块类型
        blocked_stats: dict[str, int] = Counter()
        for t in state.tiles:
            if not t.free:
                blocked_stats[t.tile_type] += 1

        # === 槽满情况：仅复活可用 ===
        if buffer_full:
            if not self.revive_used:
                # 确认无三消可能
                if not any(c >= MATCH_COUNT for c in buffer_counts.values()):
                    return "revive"
                # 有三消可能但还没触发，正常流程会处理，不需要道具
                return None
            else:
                # 槽满 + 复活已用 = 无解，只能重开
                return None

        # === 槽未满：推荐普通道具 ===

        # 规则1：缓冲槽中类型太分散（≥5种）→ 建议移出
        if unique_in_buffer >= 5:
            return "move_out"

        # 规则2：缓冲槽接近满（≥6）→ 建议移出
        if len(buffer) >= 6:
            return "move_out"

        # 规则3：缓冲中有即将成对的，但同类型都被压住了 → 建议洗牌
        for t_type, cnt in buffer_counts.items():
            if cnt >= 2:
                free_same = sum(1 for t in free_tiles if t.tile_type == t_type)
                if free_same == 0 and blocked_stats.get(t_type, 0) > 0:
                    return "shuffle"

        # 规则4：大量方块被阻塞（>60%）→ 建议洗牌
        blocked_count = sum(1 for t in state.tiles if not t.free)
        if blocked_count > len(state.tiles) * 0.6:
            return "shuffle"

        return None


def format_solution(result: SolverResult) -> str:
    """格式化求解结果为可读字符串"""
    if result.success:
        lines = [f"✅ {result.message}"]
        for i, (tile_id, tile_type) in enumerate(result.moves):
            lines.append(f"  {i + 1}. 点击方块 #{tile_id} [{tile_type}]")
        return "\n".join(lines)
    else:
        msg = f"❌ {result.message}"
        if result.recommended_item:
            item = ITEMS.get(result.recommended_item)
            if item:
                msg += f"\n💡 建议使用道具: {item.name} - {item.description}"
        return msg


if __name__ == "__main__":
    # 简单自测
    from detector import Tile

    # 构造测试数据
    test_tiles = []
    for i in range(9):
        t = Tile(i, f"type_{i % 3}", (10 * i, 10, 40, 40))
        t.free = True
        test_tiles.append(t)

    solver = SheepSolver()
    result = solver.solve(test_tiles)
    print(format_solution(result))
