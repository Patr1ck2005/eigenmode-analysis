# Agent instructions

For tasks in the personal research system, read the [canonical MyPhysics entry](D:/Obsidian/MyPhysics/System/README.md) and follow its pointers to the owning scientific project. This does not change the domain-neutral package contract. [Repository homepage](README.md); [research-system map](D:/Obsidian/MyPhysics/System/Research-System-Map.md).

This package provides reusable, domain-neutral eigenmode numerics.

## Boundary

- Runtime dependencies are NumPy, SciPy, and the Python standard library.
- Do not import plotting backends, DataFrames, simulator APIs, manifests, or
  consumer applications from `src/eigenmode_analysis/`.
- Public inputs and outputs are arrays and typed models with explicit identity,
  coordinates, traversal, and diagnostics.
- Missing or rejected modes remain missing; do not silently substitute a mode.

## Development

1. Characterize a numerical requirement with synthetic and consumer-side tests.
2. Add the schema-neutral implementation and package tests here.
3. Update consumer adapters and provenance after the shared contract is tested.
4. Use stable wheels in consumer environments; editable installs are limited to
   an isolated shared-development environment.
5. Update the public API or contract document for every release.
6. Run the package suite and dependency-boundary checks before tagging.


<!-- project-hub:integration:start -->
## 项目总览接入约定

- 本项目的开发计划与验收证据继续保存在原有项目文档中，无需为总览维护额外进度摘要。总览直接读取本地 Git 提交；不再要求更新 `PROJECT_STATUS.md`，也不删除其他会话留下的状态原文。
- 若本项目已接入主要网页，入口以根目录 `PROJECT_WEB.json` 和 `WEB_ENTRY.md` 为准。修改网页入口、启动命令、依赖、端口或停止方式时，同步更新配置、统一启动脚本及入口说明，并验证启动、打开和停止。
- 使用 `start_web.bat`、`stop_web.bat`，需要加载后端或构建改动时使用 `start_web.bat restart`。开关由 Project Hub 统一管理；不要另起重复服务、按端口直接杀进程或自动换端口。
- Git 提交标题准确描述实际改动，不把提交活动视为完成度或验收证明。更新入口不构成提交授权；每次 commit 仍须向用户确认来源分类和具体模型，遵循既有 Origin 规则。
- 多会话修改前重新读取相关文件，只修改自己负责的内容；保留个人关注和备注。已有会话需重新读取本段，本约定不会自动同步会话或发送任务。
<!-- project-hub:integration:end -->
