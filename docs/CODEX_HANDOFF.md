# TEM 软件 Codex 接手说明

## 1. 版本与项目来源

- 项目名称：矿井瞬变电磁 TEM 水害智能识别处理系统 V1.0
- 状态：工程研究原型；**不是**通过实测钻探验证、可自动判定老空水或授权开采的系统。
- 当前工程位于仓库根目录；既有代码、测试和示例数据完整保留，见 `README.md`。
- 论文参考：Yuan 等（2026），*MTDLPy: An open-source deep learning inversion software for magnetotellurics*, Computers & Geosciences, 215, 106219。与本项目的 TEM 方法**不相同**，只借鉴模块化 API、图形工作流与可扩展训练机制。

## 2. 工程关键入口

- `mine_tem/model.py`：设置、数据校验和结果对象。
- `mine_tem/io.py`：长表 CSV、宽表、中国语表头及批量读入。
- `mine_tem/processing.py`：时域衰减 QC、理想中心回线晚期近似变换、视深度代理、伪剖面、候选异常。
- `mine_tem/simpeg_engine.py`：可选 SimPEG 地表 1D 分层正反演 + SciPy 约束拟合，尚未完成依赖级验证。
- `mine_tem/learn.py`：PyTorch 配对样本监督训练和预测；**没有**可用于现场矿井分类的已验证预训练模型。
- `mine_tem/gui.py` / `run_gui.py`：PySide6 中文 GUI（原环境未能启动验证）。
- `mine_tem/cli.py`：无需 GUI 的批处理。
- `mine_tem/export.py`：CSV/PNG/NPZ/JSON 结果与溯源。
- `example/demo_raw.csv`：合成数据，不能用于宣称水害准确率。

## 3. 基线验证记录（交接时实际再次执行）

- Python 3.13.5 运行 `python -m pytest -q`，得到 `9 passed in 3.17s`（交接打包时执行；以后环境结果可能变化）。
- 历史示例处理：43 测点 × 33 门 = 1,419 行；原测试报告记录 1,415 有效门；输出演示二维伪剖面与低阻候选区，详见 `测试与交付说明.md`。
- **尚未验证**：GUI 在 Windows 启动交互；SimPEG 依赖级物理正/反演；实测设备关断波形校正；井下全空间含金属支护正/反演；二维/三维严格数值成像；真实含水/非含水盲测准确率；Windows EXE 安装包。
- 新增 `.github/workflows/ci.yml` 是给后续 GitHub 使用的 CI 配置，**尚未在 GitHub 实际执行**。

## 4. 核心风险（不许跨越）

1. 半空间晚期近似公式只适用于明确简化假设；不是井下巷道全空间电磁场的通用解。
2. 视深度是代理量，基于它计算的候选区面积不是实际含水面积，更不是储水体积。
3. 电性低阻成因非唯一，泥岩/粘土/金属构件/矿化带可能产生误判。
4. 智能网络需成对 TEM 波形与经独立核验的地下电阻率/水害标签。纯合成数据无法证明现场泛化。
5. 参考 MT 论文的 TE/TM 频域相位不是本系统 TEM 的输入；不可冒充同一物理法。

## 5. P0 启动命令

```shell
python -m pip install -e .
python -m pytest -q
python -m mine_tem.cli --demo --output output_demo
python -m mine_tem.gui
```

推荐使用 Python 3.12 专用虚拟环境；在无 Qt/图形支持时运行 CLI 并记录限制。若需学习功能安装 `torch`，若需地表 1D 物理模块安装 `simpeg` 的受支持版本并校验 API。

## 6. 交接期望

请在同一仓库继续实现，在每个里程碑后提交可审查的代码变更、测试证据和中文变更记录。保持安全限制，在有标定和钻探真值之前只能输出“低阻候选异常”。
