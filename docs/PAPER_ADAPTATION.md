# 参考论文与 TEM 软件之间的可复用边界

参考：Chongxin Yuan, Kunpeng Wang, Qilin Huang, “MTDLPy: An open-source deep learning inversion software for magnetotellurics,” *Computers & Geosciences*, 215 (2026), 106219, DOI: 10.1016/j.cageo.2026.106219。用户提供的论文 PDF 位于 `reference_paper/MTDLPy_2026_user_supplied.pdf`，仅供内部研究，请勿未经许可公开再分发。

## 论文直接支持借鉴的思路

- 第 2–4 页：三层设计——核心数据处理与模型、Python API、基于 Qt 的 GUI；主流程以标签页组织。
- 第 3–7 页：多通道输入、对数归一化、基于空间和频率维度重采样、U-Net/DinkNet 等网络模型的可插拔封装；训练日志、早停、最优权重与预测导出。
- 第 6–8 页：通过成对**正演响应—地电阻率模型**构建训练样本，而非只凭异常颜色分类。
- 第 9–13 页：合成场景基准、传统反演对照、现场公开数据示范和正演响应对比；评估不能只看图片好看。

## 不得照搬的内容

1. 论文是 MT：自然场、频域表观电阻率与相位、TE/TM 模式；矿井 TEM 是人工源、关断后时域衰减响应，输入与场方程不同。
2. 论文 32×32 MT 数据布局、0.05–320 Hz 及 TE/TM 相位通道不是 TEM 标准；应由真实 TEM 时间门与采集几何决定。
3. 论文结果指标来自 MT 数据，**不得**用来宣称本 TEM 系统的误差、精度、训练速度或水害准确率。
4. 低阻异常与实际含水之间没有一一对应的可靠关系，必须引入钻孔与地质标签。

## 开源框架应用

- SimPEG 用作物理正演/反演主要候选工具，先从严格界定的地表 1D 分层模型基准核对。
- PyTorch 用于学习模块的训练与可替换模型。
- NumPy/SciPy/pandas/Matplotlib/PySide6 用于数值、处理和界面。
- empymod 可以评估分层介质时域计算适用性，但目前**尚未集成**。
- 第三方许可证与引用请阅读 `OPEN_SOURCE.md` 并自行核查目标版本。
