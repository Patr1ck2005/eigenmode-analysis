# Eigenmode Analysis

[English](README.md) · [简体中文](README.zh-CN.md) · [合成图集](docs/assets/showcase/gallery.html) · [公开 API](docs/public-api.md)

跟踪本征模候选，保留原始身份与缺失样本。基于数组的路径／有向图跟踪明确返回指派代价、覆盖率、被拒候选和图诊断。

![相同观测在跟踪前后的对比](docs/assets/showcase/hero.png)

**实际计算的合成示例。** 两图包含相同的 265 个观测；8 个缺失槽位仍然缺失。颜色表示数值轨迹，不代表唯一物理模式身份。[源码](examples/synthetic_tracking.py) · [来源记录](docs/assets/showcase/manifest.json)。

## 主要能力

| 能力 | 公开接口 | 用途 |
|---|---|---|
| 有序扫描 | `CandidateSet`、`TrackingConfig`、`track_path` | 候选重排、分支交叉和观测缺失 |
| 参数图 | `TraversalEdge`、`rectilinear_traversal`、`track_graph` | 明确遍历、入边诊断与空站点 |
| 场与模式分析 | `symmetry`、`polarization`、`topology`、`propagation` | 数组上的可复用数值运算 |
| 采样和特征分析 | `grid_sampling`、`lineshapes`、`peak_tracking`、`field_regimes` | 分析独立于读取和渲染 |

运行依赖为 NumPy、SciPy。数值库不导入绘图后端、DataFrame、仿真器 API 或消费端。读取、物理筛选、输出路径与解释属于调用方。

```mermaid
flowchart LR
    A[数组与候选 ID] --> B[明确的路径或图]
    B --> C[按配置进行指派]
    C --> D[轨迹 ID 与数组]
    C --> E[缺失槽位、代价、诊断]
    D --> F[调用方分析或绘图]
```

## 发行线与安装

| 公开分支 | 版本 | 兼容关系 |
|---|---|---|
| 默认 `public-release` | **0.2.10** | 稳定路径／图 API，兼容 Plot Workflows 0.1.7 |
| `feature/band-tracking-v2` | **0.3.1** | 新增可选 `band_grouping`，不满足 Workflows 0.1.7 的 `<0.3` 约束 |

本检出介绍两条发行线。图集使用 **0.2.10 wheel**，不作为另一分组算法的验证。下面的固定公开源码安装需要 Python 3.10+ 和 Git，不假定存在包索引发行或版本标签。

```bash
python -m venv .venv
# 按当前 shell 的方法激活 .venv，然后执行：
python -m pip install "eigenmode-analysis @ git+https://github.com/Patr1ck2005/eigenmode-analysis.git@e172495aa0c561cf0edf251ae9828dd5e54d2d64"
python -m pip check
```

也可安装已核验的本地 `eigenmode_analysis-0.2.10-py3-none-any.whl`。稳定消费端应安装 wheel，不注入兄弟仓库源码路径。

## 可直接运行的简例

```python
import numpy as np
from eigenmode_analysis.tracking import CandidateSet, TrackingConfig, track_path

t = np.linspace(-1, 1, 21)
points = []
for i, value in enumerate(t):
    order = [1, 0] if i % 2 else [0, 1]
    values = np.array([1 + 0.2 * value, 1 - 0.2 * value])
    points.append(CandidateSet(order, values[order, None]))
result = track_path(points, coordinates=t,
                    config=TrackingConfig(n_tracks=2, derivative_weights=2))
assert result.coverage == 1.0
print(result.features.shape)  # (21, 2, 1)
```

候选 ID 是站点内的不透明标记。特征尺度与连续性假设会影响指派；代价不是概率。缺失或被拒的指派保留为 `None`／NaN，不静默替换成另一个模式。

## 实际生成的功能示例

| 交叉和缺失样本 | 有向图诊断 |
|---|---|
| ![带缺口的轨迹](docs/assets/showcase/crossing-missing.png) | ![图覆盖率与代价](docs/assets/showcase/graph-diagnostics.png) |

[路径代价诊断](docs/assets/showcase/path-diagnostics.png) · [离线 PNG／SVG 图集](docs/assets/showcase/gallery.html)。

在本仓库中安装示例绘图依赖并运行：

```bash
python -m pip install "plot-foundation @ git+https://github.com/Patr1ck2005/plot-foundation.git@c3d3cc23b2bb4da7ade7e5edf553d4a36d19db7f"
python examples/synthetic_tracking.py --out docs/assets/showcase
```

示例使用固定种子、公开 API 和独立数学定义。断言检查观测保留、预期交叉、8 个缺失槽位、空网格站点及输入顺序不变性。PNG／SVG、HTML 图集与 manifest 记录源码哈希、版本、尺寸及检查结果。

## 0.3.1 的进阶分组

请在单独环境中安装此发行线，不要混入稳定 Plot Workflows 环境：

```bash
python -m venv .venv-grouping
# 按当前 shell 的方法激活 .venv-grouping，然后执行：
python -m pip install "eigenmode-analysis @ git+https://github.com/Patr1ck2005/eigenmode-analysis.git@d82251086f50cb1428c35f457631f74c10b5afc2"
python examples/synthetic_grouping.py --out .local/advanced-grouping
```

![独立的 0.3.1 合成分组示例](docs/assets/grouping-example/grouping.svg)

[独立脚本](examples/synthetic_grouping.py)直接定义 7 × 7 的乱序无量纲复数网格，`z = 1 ± 0.25u + 0.03v + 0.02i`，并留出一个空站点。全部 **96 个观测**逐一保留 `(site, local_index)` 身份、原始 ID、payload 与源数值。6 个站点各含两个等频率观测，两个副本仍是独立观测。SVG 展示实际观测支撑和一个固定 v 截面，不增加拟合点或插值点。

运行将原始／分组 CSV、分配表、引用原始行的三角形、完整诊断与 manifest 写入新的输出目录。[预览数据与诊断](docs/assets/grouping-example/manifest.json)分别记录包版本 **0.3.1** 和算法版本 **stable-v1.0.0**。断言检查观测保留、多重性、缺失支撑及源字节不变，不认证唯一物理身份。诊断保持 `conditional_or_provisional`，没有校准概率。仅需要 NumPy、SciPy 与标准库。

请阅读[分组合同](docs/band-grouping.md)，包括一维显式种子限制和未解决情况诊断。不要隐式升级要求 `<0.3` 的工作流环境。

## 文档与开发

[公开 API](docs/public-api.md) · [架构](docs/architecture.md) · [变更记录](CHANGELOG.md) · [Agent 规则](AGENTS.md) · [四库示例](https://github.com/Patr1ck2005/plot-workflows#run-the-end-to-end-example)。

开发使用隔离环境，安装 `.[dev]` 并运行 `python -m pytest -q`。示例断言覆盖范围小于完整包测试。

[原集合概念图](docs/assets/portfolio/scientific-tools-research-v2.png)、[分组概念图](docs/assets/portfolio/band-grouping-research-v2.png)及[来源记录](docs/assets/portfolio/manifest.json)保留为辅助示意，不作为计算结果。采用 MIT 许可证；示例不声称构成实验、全波或未公开研究成果。
