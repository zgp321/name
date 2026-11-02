# 矿井瞬变电磁 TEM 水害智能识别处理系统 V1.0

> 当前交付：**可以执行的工程研究原型 / 辅助判读系统**，不是已通过矿山安全认证的井下水害自动判定仪器。没有提供未经验证的“水害识别准确率”。

最新修复：预测数值校验、质控缺口保留、最低 SNR 生效、大间距候选分段、同模型重训结果失效、导出失败回滚。验证与兼容说明见 [六项逻辑修复记录](docs/LOGIC_FIXES.md)。

## 主要内容

使用 Python + NumPy/SciPy + PySide6 实现四个侧边导航工作区：① 数据与测线；② 处理参数；③ 结果判读；④ 模型训练。界面采用深色导航、浅色卡片、质量统计和可缩放图表，支持批量导入、设备参数与阈值设置、中心回线晚期视电阻率伪剖面、低阻候选区、2.5D 示意以及 CSV/PNG/NPZ/JSON 导出。可选 PyTorch 1D-CNN 用于成对监督训练与载入预测（训练日志、验证损失、最佳模型和早停）。

同时提供命令行，允许批处理；可选 SimPEG 1D 地表分层介质正演 + SciPy 最小二乘拟合模块。

### 方法适用性（必须阅读）

1. **默认快速模式不是 2D/3D 数值反演**。它在理想圆形中心回线、地表半空间、理想阶跃关断的晚期条件下，采用解析变换计算视电阻率：

   `|dBz/dt| = μ0 M/20 * [μ0/(π ρ)]^(3/2) * t^(-5/2)`，其中 `M = I * A_tx * N_tx`，接收电压 `V = -N_rx * A_rx * dB/dt`。

   视深度仅为 `z_proxy = depth_factor * sqrt(ρ_a t/(2 μ0))`。各测点视电阻率-视深度曲线拼接成**伪剖面**。高度不确定时，深度不应用于超前探测进尺设计或水量计算。
2. **井下巷道属于全空间问题**，金属支护、关断电流、偏移距、耦合、非圆形发射线圈和磁性介质会显著改变响应。默认半空间算法不能用来直接判定老空积水，也不能自动修复任意金属干扰。所谓“修复”仅对少数孤立尖峰插值并标记；大范围干扰直接排除。
3. **SimPEG 可选模块**只适用于指定模型假设的**地表中心回线 1D 分层半空间**。多个单点 1D 反演拼接为剖面，不代表严格 2D/3D TEM 反演，也不是井下全空间反演。当前环境未安装 SimPEG，故此接口须在目标运行环境中进行进一步验证。
4. **低阻候选区不等同于水害区域**。泥岩、黏土、石墨、金属支护、矿化带、含盐水等均可能低阻。输出全部标为“待结合地质/钻探/其他物探验证”，禁止用于直接下达采掘放行指令。
5. **学习模块默认没有预训练模型**。训练前必须使用同一配置的 TEM 响应—实测地层电阻率/经物理正演验证的标签对。随包合成数据只用于软件教学和单元测试，不代表真实矿井水害识别能力。

## 环境与安装

建议 Windows 10/11 或 Linux、Python **3.12**（SimPEG 近期版要求 >=3.12）。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
python -m mine_tem.gui
```

Linux/macOS：`python3.12 -m venv .venv && source .venv/bin/activate && pip install -e . && python -m mine_tem.gui`。

### 新版桌面工作流

1. 在“数据与测线”导入文件，或点击“使用合成演示”。表格只读展示前 100 行；处理始终使用全部数据。选中单元格后按 `Ctrl+C` 复制。
2. 选择顶部的当前测线，进入“处理参数”，核对线圈面积、匝数、电流、噪声底等；点击“运行当前测线”。参数页可滚动，支持恢复默认值。
3. 在“结果判读”检查有效/拒绝时间门和候选区，使用图表工具栏缩放、平移和复位。图表与候选表可拖动调整比例；小窗口可滚动查看完整内容。数值在表格中简化显示，悬停可查看完整值，导出精度不变。
4. 点击“导出完整成果”，选择父目录。成果保存在其下的 `TEM_测线名` 子目录；不安全字符或过长名称会转换为可读前缀加确定性摘要，以区分不同测线，文件内保留原始测线名。同一测线再次导出至原目录会更新成果文件；目录属于另一测线、包含未知文件或符号链接时会拒绝写入，须改选输出位置。
5. “模型训练”需额外安装 PyTorch。缺少 PyTorch / SimPEG 时界面会说明原因，基础快速处理仍可使用。

切换测线、重新导入、修改参数或引擎后，旧结果会清除，须重新处理后才能导出。导入、计算、训练、推理、导出同时只能运行一项；任务运行时可浏览页面和日志，不能改参或关闭窗口。失败原因可展开查看，详情也记录在“模型训练”页的运行日志。导入失败会保留此前的数据及结果。

界面改版记录与验证边界见 [UI_REDESIGN_CHANGELOG.md](docs/UI_REDESIGN_CHANGELOG.md)。

### 训练模型板块

1. 选择成对 NPZ，点击“检查训练数据”，查看样本数、时间门、深度层、轴范围和标签电阻率范围。检查与实际训练共用校验：信号和标签须为有限正数；时间轴须有限、正且严格递增；深度轴须有限、非负且严格递增。改变数据路径后须重新检查，实际训练还会重新读取并验证数据。
2. 通过“选择保存位置”设置输出前缀并指定最大轮次，开始训练。保留原有 1D-CNN、8:2 划分、学习率与早停策略；保存的文件名仍为 `前缀.pt`、`前缀_metadata.npz`、`前缀_history.csv`。
3. 训练时查看真实轮次、训练/验证 MSE 和损失曲线。提前结束会显示实际轮次及早停状态。MSE 是归一化对数电阻率目标空间的损失，不是现场水害识别准确率。
4. 训练完成后自动定位生成的模型，也可“选择已有模型”，自动匹配同名前缀元数据，检查门数、深度层和轴范围。此步骤**不加载权重**，只能确认元数据和配套文件；实际推理会重新验证元数据并使用现有安全权重加载流程。请只加载可信来源的模型。
5. 当前测线的时间门须与训练门序列一致，质量拒绝规则保持不变。更换用于推理的模型会清除旧推理结果，防止误导出；模型选择不会清除普通快速成像结果。

检查数据和模型元数据不需要 PyTorch。真实训练/推理需另装 `torch`。本次已在 Linux 云环境的 CPU PyTorch 2.15.0+cpu 验证短训练与推理；尚未验证 Windows/GPU。随包 `example/homogeneous_training_demo.npz` 仅用于教学，不是已验证的现场训练集。详见 [训练操作优化记录](docs/TRAINING_OPERATIONS_CHANGELOG.md)。

没有图形环境也能执行：

```bash
python -m mine_tem.cli --demo --output output_demo
python -m mine_tem.cli example/demo_raw.csv --line DEMO --output output
```

需要物理1D反演时另行安装：`pip install 'simpeg>=0.25,<0.27'`。

需要数据监督训练：`pip install torch`。训练数据生成接口仅用于格式说明：

```python
from mine_tem.training_demo import make_training_demo
make_training_demo('homogeneous_tutorial_only.npz')  # 仅用于教学的均匀半空间
from mine_tem.learn import train
train('homogeneous_tutorial_only.npz', 'my_models/lesson', epochs=20)
# 监督模型推理：from mine_tem.learn import predict_line
```

## 数据格式

表头（长表）：`line_id,station,x_m,y_m,elevation_m,time_s,dbdt_t_s` 或 `voltage_v`。每个测点包含若干正时间门。电压为 **V**，磁场时间导数为 **T/s**，时间为 **s**。CSV / TXT / TSV / XLSX；支持 `时间ms`、`时间微秒` 和 `电压mV` / `电压uV` 中文/英文别名及 `t_0.1ms` 形式宽表。每条测线的 `x_m` 应无重复里程。二进制仪器格式需按厂家提供解析插件后才能导入。

## 结果说明

- `gates_processed.csv`：输入、质控标记、清洗后 dB/dt、视电阻率、视深度；任何拒绝/修复都有记录。
- `pseudo_section_xyz.csv`：里程、视深度、视电阻率、异常标记；用于 GIS/Surfer/ParaView。
- `anomaly_candidates.csv`：相对背景 + 绝对阈值的候选块、范围及**伪剖面面积**，不是实际含水面积，更不是三维体积。
- `decay.png`、`pseudo_section.png`、`anomalies_2p5d.png`：可审查的图形成果（2.5D 图只是二维掩膜假设延伸）。
- `manifest.json`：仪器参数、阈值、所用算法、门数统计及限制。

## 目录

```text
mine_tem/model.py       核心数据结构和安全校验
mine_tem/io.py          中文/英文表头、批量导入
mine_tem/processing.py  质量控制、TEM视电阻率、成像和异常圈定
mine_tem/simpeg_engine.py  可选 SimPEG 1D地表正反演
mine_tem/learn.py       成对标签监督训练与预测
mine_tem/gui.py         PySide6中文GUI
mine_tem/gui_window.py  新版工作台布局、交互与任务状态
mine_tem/gui_support.py 只读复制表格、样式与后台任务
mine_tem/gui_training.py 训练操作、数据概览与实时损失曲线
mine_tem/training_info.py 无需PyTorch的数据/模型元数据检查
mine_tem/cli.py         CLI批处理
mine_tem/export.py      数据/图片/报告归档
mine_tem/demo.py        可重复理想半空间模拟数据
```

## 验证

`python -m pytest -q`，包括半空间正演—反算代数一致性、时间单位、多测线、无效数据、批量导入、导出和训练模块基本流程。**这些不是现场精度验收**，正式工程还需仪器标定、井下全空间电磁模拟、盲测数据集、施工钻孔验证、误判/漏判率和第三方审查。

来源和许可证见 [OPEN_SOURCE.md](OPEN_SOURCE.md)。
