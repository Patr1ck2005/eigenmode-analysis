import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import median_abs_deviation

def print_debug(_message):
    """Compatibility no-op for the old A debug hook."""

    return None

# Selector traversal is schema-neutral and is maintained by eigenmode-analysis.
# The local names below remain as compatibility imports for older A scripts.
from .additional import (
    infer_candidate_selector as _shared_infer_candidate_selector,
    pick_candidate_data as _shared_pick_candidate_data,
    pick_candidate_data_stable as _shared_pick_candidate_data_stable,
)


# ---------- 工具函数：NaN 判定（兼容复数） ----------
def _is_nan_scalar(x):
    """兼容 float/complex 的 NaN 判定；非数值返回 False。"""
    try:
        return np.isnan(x)
    except TypeError:
        if isinstance(x, complex):
            return np.isnan(x.real) or np.isnan(x.imag)
    return False


# ---------- 一侧差分法估计导数 ----------
def estimate_derivative_one_sided_vector_component(
    Zg_out, assigned, idx, s, axis, delta, vector_component_idx, output_dtype
):
    """
    在多维数组 Zg_out (存储形状为 (m, num_vector_dims) 的 np.array) 上，
    沿 axis 方向在点 idx 处，估计第 s 条曲面在第 vector_component_idx 个向量分量上的偏导数。
    优先使用已赋值的后向差分，不可用时使用已赋值的前向差分；
    若都不可用或涉及 NaN，返回 0.0 (或 0.0+0.0j)。
    """
    idx_list = list(idx)

    current_val_s_comp = Zg_out[idx][s, vector_component_idx]
    if _is_nan_scalar(current_val_s_comp):
        return output_dtype(0.0)

    # 后向差分
    if idx[axis] - 1 >= 0:
        idx_list_b = idx_list.copy()
        idx_list_b[axis] -= 1
        nb = tuple(idx_list_b)
        if assigned[nb]:
            prev_val_s_comp = Zg_out[nb][s, vector_component_idx]
            if not _is_nan_scalar(prev_val_s_comp):
                return (current_val_s_comp - prev_val_s_comp) / delta

    # 前向差分
    if idx[axis] + 1 < Zg_out.shape[axis]:
        idx_list_f = idx_list.copy()
        idx_list_f[axis] += 1
        nf = tuple(idx_list_f)
        if assigned[nf]:
            next_val_s_comp = Zg_out[nf][s, vector_component_idx]
            if not _is_nan_scalar(next_val_s_comp):
                return (next_val_s_comp - current_val_s_comp) / delta

    return output_dtype(0.0)


# ---------- 伴随数据：选择器（selector）机制 ----------
def _infer_additional_selector(ad_cell, num_candidates, resolve="deepest"):
    """
    在一个参考单元 ad_cell 上“定位”候选维的层级/轴，并返回 selector（若干个 '*' 组成的 tuple）：
      ()        -> 当前层就是候选层（len == num_candidates）
      ('*',)    -> 下一层是候选层（对当前层每个元素取 [cand_idx]）
      ('*','*') -> 再下一层是候选层，依此类推。

    当存在多个匹配层级时依据 resolve 规则：
      - 'deepest'    : 选更深的路径（推荐，用于 [b][cand][...]）
      - 'shallowest' : 选更浅的路径
      - 'error'      : 抛出错误（严格模式）
    """
    def _len_safe(x):
        try:
            return len(x)
        except Exception:
            return None

    def _is_seq(x):
        return isinstance(x, (list, tuple, np.ndarray))

    paths = []

    def dfs(node, path):
        L = _len_safe(node)
        if L is None:
            return
        if L == num_candidates:
            paths.append(tuple(path))
        if _is_seq(node) and L > 0:
            try:
                first = node[0]
            except Exception:
                first = None
            dfs(first, path + ('*',))

    dfs(ad_cell, ())

    if not paths:
        return None

    uniq_paths = list(sorted(set(paths), key=lambda p: len(p)))
    if len(uniq_paths) == 1:
        return uniq_paths[0]

    # 多条路径 -> 依据 resolve 处理
    if resolve == "error":
        raise ValueError(
            f"additional_data 结构在参照点存在歧义：检测到多个层级长度都等于候选数，路径={uniq_paths}。"
            f"请使候选维位于固定且唯一的层级，或在调用时显式指定 additional_selector。"
        )
    elif resolve == "deepest":
        return uniq_paths[-1]   # 选更深的（例如优先 ('*',) 而不是 ()）
    elif resolve == "shallowest":
        return uniq_paths[0]    # 选更浅的
    else:
        # 未知策略，回退为最深
        return uniq_paths[-1]


def _pick_additional_by_selector(ad_cell, cand_idx, selector):
    """
    使用 selector 稳定地抽取第 cand_idx 个候选的伴随数据。
    selector: 由若干个 '*' 组成的 tuple，表示“向下映射的层数”。
    返回：保持其它轴/维的结构；若遇到 None/越界，返回 None。
    """
    if ad_cell is None or selector is None:
        return None

    def map_down(node, depth=0):
        if depth == len(selector):
            # 命中候选层：在此层按 cand_idx 取元素
            try:
                L = len(node)
                if not (0 <= cand_idx < L):
                    return None
                return node[cand_idx]
            except Exception:
                return None

        token = selector[depth]
        if token != '*':
            return None

        # 这一层需要对每个元素 map
        try:
            L = len(node)
        except Exception:
            return None

        out = []
        for i in range(L):
            try:
                sub = node[i]
            except Exception:
                sub = None
            out.append(map_down(sub, depth + 1))
        return out

    return map_down(ad_cell)


def _pick_additional_candidate_stable(ad_cell, cand_idx, selector):
    """封装一层：selector 缺失则直接返回 None（宁缺毋滥，避免误取）。"""
    return _pick_additional_by_selector(ad_cell, cand_idx, selector) if selector is not None else None


# Keep historical private names import-compatible while making the shared
# implementation the runtime owner for all new and legacy call sites.
_infer_additional_selector = _shared_infer_candidate_selector
_pick_additional_by_selector = _shared_pick_candidate_data
_pick_additional_candidate_stable = _shared_pick_candidate_data_stable


# ---------- 主函数：匈牙利分组 + 单边导数连续性 ----------
def group_vectors_one_sided_hungarian(
    Z_vector_components,   # list[np.ndarray], 每个元素 dtype=object, shape=dims；单元里是“候选列表”
    deltas,                # tuple[float], len=n_dims
    value_weights,         # np.ndarray, shape=(n_dims, n_dims)
    deriv_weights,         # np.ndarray, shape=(n_dims, n_dims)
    max_m=None,
    initial_derivatives=None,  # (max_m, n_dims, num_vector_dims) 未使用可为 None
    nan_cost_penalty=1e9,
    additional_data=None,      # dtype=object，与 Z 前 n 维对齐；后续维度任意（如 [b][cand][...]）
    additional_selector=None,  # tuple，如 (), ('*',), ('*','*')；显式指定候选层（推荐）
    selector_resolve="deepest",  # 'deepest'|'shallowest'|'error'：自动推断多解时的策略
    auto_split_streams: bool = False,  # 是否启用自动拆分
    min_segment_points: int = 3,  # 子段最小点数（< 此值剔除）
    mad_multiplier: float = 5.0,  # 异常检测倍数
    freq_rank_penalty: float = 0.0,  # freq rank 约束权重：stream s 倾向选 rank s 候选（0=最低频）
    freq_window: float = 0.0,  # freq 连续性窗口：候选 freq 与前一点差超过此值时排除（防跳到另一条 band）
):
    """
    在任意 n 维网格上，对"候选列表"进行分组（匈牙利算法 + 单边导数连续性）。

    算法 4 层机制（详见 docs/agent-plotting/protocols/band-tracking-hungarian.md）：
      1. 字典序遍历 + origin 排序（起点按 real 排序赋 stream 0..max_m-1）
      2. 匈牙利全局优化（per-cell 代价矩阵 C_total = Σ_neighbors(value+deriv cost)）
      3. NaN 候选代价 mask（np.where isfinite，防 scipy linear_sum_assignment NaN 行为）
      4. anchor freq 约束（freq_window>0 时，stream 维护最近有效 freq 跨越 NaN 保持，
         候选 freq 与 anchor 差 > freq_window 代价设 INF_COST，分配后不可靠置 NaN）

    主数据：Z_vector_components 决定；每格点若干候选（list），每候选是 num_vector_dims 维向量。
    伴随数据：additional_data 与网格对齐；单元内某层是"候选维"，由 additional_selector 指定。

    关键参数：
      - nan_cost_penalty: NaN stream/候选代价。禁用 0（NaN stream 零代价=随机初始化）
      - freq_window: freq 连续性窗口。>0 启用 anchor 约束，防止 stream 跳到另一条物理 band。
        真实数据推荐 0.005。mode 消失时 stream NaN（诚实断带），不跳到别的 band
      - freq_rank_penalty: per-cell rank 约束权重。排除高频干扰有效，但对候选消失场景无效
        （per-cell rank 在候选消失时偏移）。4a 不用
      - auto_split_streams: 后处理连通分量拆分。治标不治本（丢弃跳变产生 NaN），
        freq_window 已在根源防跳变。仅确需分离多条 band 且接受 NaN 时用
      - mad_multiplier: auto_split 异常检测倍数。5.0=激进，20.0=保守

    返回：
      - additional_data 为 None -> 仅返回 Zg_out
      - 否则 -> 返回 (Zg_out, additional_grouped)
    """
    if not Z_vector_components:
        raise ValueError("Z_vector_components cannot be empty.")

    dims = Z_vector_components[0].shape
    n_dims = len(dims)
    num_vector_dims = len(Z_vector_components)
    origin = tuple([0] * n_dims)

    # 校验 additional_data 的前 n_dims 维
    if additional_data is not None:
        if additional_data.shape[:n_dims] != dims:
            raise ValueError(
                f"additional_data 前 {n_dims} 维应与 Z 的 dims 一致：{dims}，但得到 {additional_data.shape[:n_dims]}"
            )

    # 确定输出 dtype（float/complex）
    output_dtype = float
    for idx in np.ndindex(*dims):
        if Z_vector_components[0][idx] is not None and len(Z_vector_components[0][idx]) > 0:
            if isinstance(Z_vector_components[0][idx][0], complex):
                output_dtype = complex
            break

    # 确定追踪的 m
    if max_m is None:
        m = 0
        has_data = False
        for idx in np.ndindex(*dims):
            if Z_vector_components[0][idx] is not None:
                m = max(m, len(Z_vector_components[0][idx]))
                if len(Z_vector_components[0][idx]) > 0:
                    has_data = True
        if not has_data:
            raise ValueError("Z_vector_components appears to contain no data for any point.")
        if m == 0:
            raise ValueError("Z_vector_components contains data points, but all lists are empty.")
    else:
        m = max_m

    # 输出容器
    Zg_out = np.empty(dims, dtype=object)
    assigned = np.zeros(dims, dtype=bool)
    additional_grouped = np.empty(dims, dtype=object) if additional_data is not None else None

    # --- 确定 additional_data 的 selector（一次性，保证全局一致） ---
    ad_selector = None
    if additional_data is not None:
        if additional_selector is not None:
            # 用户显式指定的 selector：最稳定、推荐
            ad_selector = tuple(additional_selector)
        else:
            # 自动推断（遇多解按策略处理；默认 'deepest'，可避免 [(), ('*',)] 的常见歧义）
            # 参照格点：优先 origin，否则找第一个有候选的格点
            ref_idx = origin
            ref_num_candidates = 0
            if Z_vector_components[0][origin] is not None:
                ref_num_candidates = len(Z_vector_components[0][origin])
            if ref_num_candidates == 0:
                for _idx in np.ndindex(*dims):
                    if Z_vector_components[0][_idx] is not None and len(Z_vector_components[0][_idx]) > 0:
                        ref_idx = _idx
                        ref_num_candidates = len(Z_vector_components[0][_idx])
                        break
            if ref_num_candidates > 0:
                ad_selector = _infer_additional_selector(
                    additional_data[ref_idx], ref_num_candidates, resolve=selector_resolve
                )
            # 若 ad_selector 为 None，则后面取伴随数据会返回 None（宁缺毋滥）

    # --- 处理原点（按值排序，作为起点） ---
    origin_num_candidates = 0
    if Z_vector_components[0][origin] is not None:
        origin_num_candidates = len(Z_vector_components[0][origin])

    if origin_num_candidates == 0:
        Zg_out[origin] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        if additional_grouped is not None:
            additional_grouped[origin] = [None] * m
    else:
        idxs = list(range(origin_num_candidates))
        idxs.sort(key=lambda ii: (
            (Z_vector_components[0][origin][ii].real
             if isinstance(Z_vector_components[0][origin][ii], complex)
             else Z_vector_components[0][origin][ii])
        ))

        sorted_origin_vectors = []
        for ii in idxs:
            vec = [Z_k[origin][ii] for Z_k in Z_vector_components]
            sorted_origin_vectors.append(np.array(vec, dtype=output_dtype))

        temp_arr = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        num_to_fill = min(m, len(sorted_origin_vectors))
        if num_to_fill > 0:
            temp_arr[:num_to_fill, :] = np.array(sorted_origin_vectors[:num_to_fill])
        Zg_out[origin] = temp_arr

        if additional_grouped is not None:
            ad = additional_data[origin]
            ordered_ad = [None] * m
            for s in range(num_to_fill):
                cand_idx = idxs[s]
                picked = _pick_additional_candidate_stable(ad, cand_idx, ad_selector)
                ordered_ad[s] = picked
            additional_grouped[origin] = ordered_ad

    assigned[origin] = True

    # 预计算权重和（值连续性代价中 norm 不依赖于 j）
    w_val_sum = value_weights.sum(axis=1)  # (n_dims,)

    # stream anchor freq：每个 stream 最近有效 freq，用于 freq_window 约束。
    # 即使中间有 NaN（mode 消失），anchor 保持上一个有效值，防止 stream NaN 后
    # 重新初始化时跳到另一条 band。
    stream_anchor_freq = np.full(m, np.nan)
    if freq_window > 0 and num_vector_dims > 0 and Zg_out[origin] is not None:
        for s in range(m):
            v0 = np.real(Zg_out[origin][s, 0])
            if np.isfinite(v0):
                stream_anchor_freq[s] = v0

    # --- 按字典序遍历网格其余点 ---
    for idx in np.ndindex(*dims):
        if assigned[idx]:
            continue

        # 已赋值的后向邻点
        neighbors = []
        for d in range(n_dims):
            if idx[d] > 0:
                nb = list(idx); nb[d] -= 1; nb = tuple(nb)
                if assigned[nb]:
                    neighbors.append((nb, d))

        if not neighbors:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            if additional_grouped is not None:
                additional_grouped[idx] = [None] * m
            assigned[idx] = True
            continue

        # 当前格点候选
        num_candidates = 0
        if Z_vector_components[0][idx] is not None:
            num_candidates = len(Z_vector_components[0][idx])

        if num_candidates == 0:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            if additional_grouped is not None:
                additional_grouped[idx] = [None] * m
            assigned[idx] = True
            continue

        # 组装候选向量矩阵 (num_candidates, num_vector_dims)
        candidates_mat = np.array([
            [Z_k[idx][i] for Z_k in Z_vector_components]
            for i in range(num_candidates)
        ], dtype=output_dtype)

        ad_candidates = additional_data[idx] if additional_data is not None else None

        # 代价矩阵 (m, num_candidates)
        C_total = np.zeros((m, num_candidates), dtype=float)

        for neighbor, grow_dir in neighbors:
            v_prev_arr = Zg_out[neighbor]  # (m, num_vector_dims)
            stream_all_nan = np.all(np.isnan(v_prev_arr), axis=1)  # (m,) bool

            # 预计算导数所需的邻点索引（per (neighbor, d)，不随 s/k 变化）
            nb_bwd = [None] * n_dims
            nb_fwd = [None] * n_dims
            for d in range(n_dims):
                if neighbor[d] > 0:
                    bwd = list(neighbor); bwd[d] -= 1; bwd = tuple(bwd)
                    if assigned[bwd]:
                        nb_bwd[d] = bwd
                if neighbor[d] + 1 < dims[d]:
                    fwd = list(neighbor); fwd[d] += 1; fwd = tuple(fwd)
                    if assigned[fwd]:
                        nb_fwd[d] = fwd

            # 内联导数估计（消除 12 万次函数调用开销）
            d_prev = np.zeros((m, n_dims, num_vector_dims), dtype=output_dtype)
            for s in range(m):
                if stream_all_nan[s]:
                    continue
                for d in range(n_dims):
                    delta = deltas[d]
                    bwd = nb_bwd[d]
                    fwd = nb_fwd[d]
                    for k in range(num_vector_dims):
                        current = Zg_out[neighbor][s, k]
                        if np.isnan(current):
                            continue
                        deriv = output_dtype(0.0)
                        if bwd is not None:
                            prev_val = Zg_out[bwd][s, k]
                            if not np.isnan(prev_val):
                                deriv = (current - prev_val) / delta
                            elif fwd is not None:
                                next_val = Zg_out[fwd][s, k]
                                if not np.isnan(next_val):
                                    deriv = (next_val - current) / delta
                        elif fwd is not None:
                            next_val = Zg_out[fwd][s, k]
                            if not np.isnan(next_val):
                                deriv = (next_val - current) / delta
                        d_prev[s, d, k] = deriv

            # 向量化代价计算
            C_nb = np.full((m, num_candidates), fill_value=nan_cost_penalty * 2, dtype=float)

            for s in range(m):
                if stream_all_nan[s]:
                    C_nb[s, :] = nan_cost_penalty
                    continue

                v = v_prev_arr[s]  # (num_vector_dims,)
                # broadcast 差值：所有候选 vs 当前流
                diffs = candidates_mat - v  # (num_candidates, num_vector_dims)

                # 值连续性代价（L2² 天然打破同侧简并）
                cost_val = np.sum(np.abs(diffs)**2, axis=1) * w_val_sum[grow_dir]

                # 导数连续性代价（L2² 在真交叉处打破简并）
                cost_der = np.zeros(num_candidates, dtype=float)
                for j in range(n_dims):
                    deriv_diff = diffs - d_prev[s, j, :] * deltas[j]
                    cost_der += deriv_weights[grow_dir, j] * np.sum(np.abs(deriv_diff)**2, axis=1)

                row_cost = cost_val + cost_der
                # NaN 候选（伪模被滤掉、复数含 NaN-real 等）会让 row_cost 出 NaN，
                # scipy.linear_sum_assignment 对 NaN 行为版本依赖，必须替换为有限大代价。
                # 用 2×penalty（与 C_nb 初始填充一致）保证 NaN 候选不会被随机选中。
                mask_penalty = nan_cost_penalty if nan_cost_penalty > 0 else 1.0
                C_nb[s, :] = np.where(np.isfinite(row_cost), row_cost, mask_penalty * 2)

            C_total += C_nb

        # --- freq rank 约束（让 stream s 优先选 freq rank s 的候选）---
        # 解决"max_num < 真实模式数时低频 stream 跳到高频候选"的问题。
        # rank 0 = 最低频候选。惩罚 (rank - s)^2 * weight，让低 stream 锁定低频。
        # NaN 候选 real 是 NaN，np.argsort 把 NaN 排最后（rank 最高），自然被低 stream 避开。
        if freq_rank_penalty > 0 and num_vector_dims > 0:
            candidate_freqs = np.real(candidates_mat[:, 0])
            ranks = np.argsort(np.argsort(candidate_freqs))
            for s in range(m):
                C_total[s, :] += (ranks - s) ** 2 * freq_rank_penalty

        # --- freq 连续性窗口（防止 stream 跳到另一条物理 band）---
        # 当某物理 band 的候选在某 cell 消失时，per-cell rank 会偏移（rank 1 变成另一条
        # band 的 freq），导致 stream 跳变。窗口约束：候选 freq 与 stream anchor freq
        # （最近有效值，跨越 NaN 保持）差 > freq_window 时，代价设为 INF_COST，
        # 分配后检测到不可靠则置 NaN（宁可断带也不跳到别的 band）。
        INF_COST = 1e15
        if freq_window > 0 and num_vector_dims > 0:
            candidate_freqs = np.real(candidates_mat[:, 0])
            for s in range(m):
                anchor = stream_anchor_freq[s]
                if not np.isfinite(anchor):
                    continue  # stream 从未有效（无 origin 也无后续），不约束
                out_of_window = np.abs(candidate_freqs - anchor) > freq_window
                if out_of_window.any():
                    C_total[s, out_of_window] += INF_COST

        # 匈牙利分配
        row_ind, col_ind = linear_sum_assignment(C_total)

        ordered = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        if additional_grouped is not None:
            ordered_ad = [None] * m
        # freq_window 启用时，超出窗口的分配代价 ~INF_COST，置 NaN（宁可断带也不跳）
        reliable_threshold = INF_COST / 2 if freq_window > 0 else np.inf
        for s_idx, c_idx in zip(row_ind, col_ind):
            if s_idx < m:
                if C_total[s_idx, c_idx] >= reliable_threshold:
                    continue  # 不可靠分配（超出窗口），保持 NaN，anchor 不更新
                ordered[s_idx, :] = candidates_mat[c_idx]
                if freq_window > 0 and num_vector_dims > 0:
                    new_freq = np.real(candidates_mat[c_idx, 0])
                    if np.isfinite(new_freq):
                        stream_anchor_freq[s_idx] = new_freq
                if additional_grouped is not None:
                    picked = _pick_additional_candidate_stable(ad_candidates, c_idx, ad_selector)
                    ordered_ad[s_idx] = picked

        Zg_out[idx] = ordered
        if additional_grouped is not None:
            additional_grouped[idx] = ordered_ad
        assigned[idx] = True

    # --------------------------------------------------------------
    # 【彻底修复版】后处理：每条流自动拆分（适用于任意维度网格）
    # --------------------------------------------------------------
    if auto_split_streams:
        # 1. 收集每条流的所有非 NaN 点
        stream_points = [[] for _ in range(m)]
        for idx in np.ndindex(*dims):
            for s in range(m):
                if Zg_out[idx] is None:
                    continue
                vec = Zg_out[idx][s]
                if vec is None or np.all([_is_nan_scalar(v) for v in vec]):
                    continue
                ad = additional_grouped[idx][s] if additional_grouped is not None else None
                stream_points[s].append((idx, vec.copy(), ad))

        # 2. 对每条流单独处理
        from collections import defaultdict
        for s in range(m):
            points = stream_points[s]
            if len(points) < 3:  # 太少直接跳过
                continue

            # 网格索引 -> 点索引（O(N) 一次构建，diffs 和 graph 共用）
            pos_to_i = {points[i][0]: i for i in range(len(points))}

            # 计算所有物理相邻点对的差值，同时缓存边（O(N * n_dims * 2)）
            # 原 O(N²) 全对扫描在 36k 点的 stream 上需数分钟，无法用于真实数据
            edges = []
            diffs = []
            for i, (idx1, vec1, _) in enumerate(points):
                for d in range(n_dims):
                    for direction in [-1, 1]:
                        nidx_list = list(idx1)
                        if 0 <= idx1[d] + direction < dims[d]:
                            nidx_list[d] += direction
                            nidx = tuple(nidx_list)
                            if nidx in pos_to_i:
                                j = pos_to_i[nidx]
                                diff = float(np.linalg.norm(vec1 - points[j][1]))
                                diffs.append(diff)
                                edges.append((i, j, diff))

            if len(diffs) == 0:
                continue

            diffs_arr = np.array(diffs)
            median_diff = np.median(diffs_arr)
            mad = median_abs_deviation(diffs_arr, scale='normal')  # scale='normal' ≈ 1.4826 * MAD
            if mad < 1e-12:
                mad = 1e-12

            threshold = median_diff + mad_multiplier * mad  # 自适应阈值

            # 3. 构建该流的连通分量（用预计算的 edges，不再重复计算 diff）
            graph = defaultdict(list)
            for i, j, diff in edges:
                if diff <= threshold:  # 小于阈值才连边
                    graph[i].append(j)
                    graph[j].append(i)

            # 4. DFS 提取所有连通分量（子段）
            visited = [False] * len(points)
            segments = []
            for i in range(len(points)):
                if not visited[i]:
                    component = []
                    stack = [i]
                    visited[i] = True
                    while stack:
                        node = stack.pop()
                        component.append(node)
                        for nb in graph[node]:
                            if not visited[nb]:
                                visited[nb] = True
                                stack.append(nb)
                    segments.append([points[k] for k in component])

            # 5. 清空原流
            for idx, _, _ in points:
                Zg_out[idx][s] = np.full(num_vector_dims, np.nan, dtype=output_dtype)
                if additional_grouped is not None:
                    additional_grouped[idx][s] = None

            # 6. 把足够长的子段放回去（按长度降序，最长的一段优先占原 s）
            segments = [seg for seg in segments if len(seg) >= min_segment_points]
            segments.sort(key=len, reverse=True)

            target_stream = s
            for seg in segments:
                while target_stream < m:
                    # 检查该流是否完全空
                    empty = all(
                        Zg_out[idx][target_stream] is None or
                        np.all([_is_nan_scalar(v) for v in Zg_out[idx][target_stream]])
                        for idx in np.ndindex(*dims)
                    )
                    if empty:
                        for idx, vec, ad in seg:
                            Zg_out[idx][target_stream] = vec
                            if additional_grouped is not None:
                                additional_grouped[idx][target_stream] = ad
                        print_debug(f"[Split] Stream {s} → new stream {target_stream} (length {len(seg)})")
                        target_stream += 1
                        break
                    target_stream += 1
                else:
                    print_debug(f"[Discard] Segment of length {len(seg)} discarded (no empty stream slot)")
                    break
    # --------------------------------------------------------------

    if additional_grouped is None:
        return Zg_out
    return Zg_out, additional_grouped
def ordered_vectors_by_abs(
    Z_vector_components,        # list[np.ndarray], dtype=object, shape=dims
    additional_data=None,       # dtype=object，与 Z 前 n 维对齐
    additional_selector=None,   # tuple，如 (), ('*',), ('*','*')
    selector_resolve="deepest"
):
    """
    简单分组：
    - 每个格点独立
    - 按候选“大小”排序
    - 直接填入第 0..m-1 条流
    - additional_data 与候选一一对应
    """

    if not Z_vector_components:
        raise ValueError("Z_vector_components cannot be empty.")

    dims = Z_vector_components[0].shape
    n_dims = len(dims)
    num_vector_dims = len(Z_vector_components)

    # 校验 additional_data
    if additional_data is not None:
        if additional_data.shape[:n_dims] != dims:
            raise ValueError(
                f"additional_data 前 {n_dims} 维应与 Z 的 dims 一致：{dims}"
            )

    # 推断 dtype
    output_dtype = float
    for idx in np.ndindex(*dims):
        cell = Z_vector_components[0][idx]
        if cell is not None and len(cell) > 0:
            if isinstance(cell[0], complex):
                output_dtype = complex
            break

    m = 0
    for idx in np.ndindex(*dims):
        cell = Z_vector_components[0][idx]
        if cell is not None:
            m = max(m, len(cell))
    if m == 0:
        raise ValueError("All candidate lists are empty.")

    # 输出
    Zg_out = np.empty(dims, dtype=object)
    additional_grouped = (
        np.empty(dims, dtype=object) if additional_data is not None else None
    )

    # ---------- 确定 additional selector（一次性，全局一致） ----------
    ad_selector = None
    if additional_data is not None:
        if additional_selector is not None:
            ad_selector = tuple(additional_selector)
        else:
            # 自动推断：找第一个有候选的格点
            for idx in np.ndindex(*dims):
                cell = Z_vector_components[0][idx]
                if cell is not None and len(cell) > 0:
                    ad_selector = _infer_additional_selector(
                        additional_data[idx],
                        len(cell),
                        resolve=selector_resolve
                    )
                    break
    # ---------------------------------------------------------------

    # 遍历所有格点
    for idx in np.ndindex(*dims):

        cell = Z_vector_components[0][idx]

        if cell is None or len(cell) == 0:
            Zg_out[idx] = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
            if additional_grouped is not None:
                additional_grouped[idx] = [None] * m
            continue

        num_candidates = len(cell)

        # 候选索引按“大小”排序（使用第一个分量）
        idxs = list(range(num_candidates))
        idxs.sort(
            key=lambda ii: (
                cell[ii].real if isinstance(cell[ii], complex) else cell[ii]
            )
        )

        # 填充 Z
        ordered = np.full((m, num_vector_dims), np.nan, dtype=output_dtype)
        for s, cand_idx in enumerate(idxs[:m]):
            vec = [Z_k[idx][cand_idx] for Z_k in Z_vector_components]
            ordered[s, :] = np.array(vec, dtype=output_dtype)
        Zg_out[idx] = ordered

        # 填充 additional
        if additional_grouped is not None:
            ad_cell = additional_data[idx]
            ordered_ad = [None] * m
            for s, cand_idx in enumerate(idxs[:m]):
                ordered_ad[s] = _pick_additional_candidate_stable(
                    ad_cell, cand_idx, ad_selector
                )
            additional_grouped[idx] = ordered_ad

    if additional_grouped is None:
        return Zg_out
    return Zg_out, additional_grouped
