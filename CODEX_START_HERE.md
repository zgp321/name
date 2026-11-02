# 交给 Codex 接续开发：从这里开始

这个 ZIP 包含已有 Python 源码、单元测试、演示数据、Windows 脚本、论文 PDF（仅供私人研究参考）及 Codex 的项目指令。

## 用本地 Codex 接手（推荐，不需要公开 GitHub）

1. 将这个 ZIP 解压到计算机中的一个固定文件夹，进入 `TEM_WaterRisk_V1.0_Codex_Handoff`（根目录有 `AGENTS.md` 和 `pyproject.toml`）。
2. 打开 Codex 桌面应用，选择该**本地文件夹**作为项目；或使用 VS Code 的 Codex 扩展/CLI 在该目录开启会话。
3. 将仓库根目录的 `CODEX_TASK.md` 内容粘贴给 Codex（也可以让 Codex 先读取该文件）。Codex 会自动发现根目录 `AGENTS.md` 的持续性工程指令。
4. 请 Codex 按第一阶段执行测试、修复和新增功能，审查代码差异后再继续下一阶段。

## 用 Codex Cloud 接手

1. 在 GitHub 新建**私有仓库**，将本项目代码推送进去（尤其不要在公开仓库上传专有矿山实测数据或论文 PDF）。`reference_paper/` 已配置为 gitignore，不会被普通 `git add .` 上传。
2. 进入 Codex Cloud，连接该私有仓库并准备环境；使用 `CODEX_TASK.md` 作为首次任务描述。
3. 让 Codex 按 `docs/DEVELOPMENT_ROADMAP.md` 的阶段执行，检查其生成的 PR/变更和测试。

## 本次交接的边界

当前聊天无法直接将这里的本地沙箱文件作为任务注入 Codex 的独立会话，也未擅自把源码上传到公开 GitHub 仓库。这个 ZIP 是可交接工程包，而非已在 Codex 后台启动的任务。

## 文件索引

- `AGENTS.md`：Codex 项目长期指令
- `CODEX_TASK.md`：第一次开发任务，直接复制给 Codex
- `docs/CODEX_HANDOFF.md`：现状、验证结果、路径与限制
- `docs/DEVELOPMENT_ROADMAP.md`：分阶段开发计划与验收标准
- `docs/PAPER_ADAPTATION.md`：论文可借鉴内容与不能照搬的部分
- `reference_paper/MTDLPy_2026_user_supplied.pdf`：论文（仅供本地学习）
- `README.md`：现有程序安装和使用说明
