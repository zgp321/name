# 已核对的基础项目与依赖边界

- SimPEG（https://github.com/simpeg/simpeg ，MIT）用于可选的分层地球 TEM 物理正演和数值拟合；本仓库仅调用其开放 API，不复制源码。注意 SimPEG 最新版需要 Python >=3.12。
- empymod（https://github.com/emsig/empymod ，Apache-2.0）是可选的后续分层电磁正演扩展参考，**V1.0 未集成其运行接口**。
- PyGMI（https://github.com/Patrick-Cole/pygmi ，GPL-3.0-or-later）有可参考的 GUI 和 TEM 1D 反演能力；本工程**没有复制**其 GPL 源码。
- MTDLPy（论文 2026, Computers & Geosciences, DOI:10.1016/j.cageo.2026.106219）仅作为三层模块化、可视化训练流程的概念参考；**MT 核心反演绝非 TEM 反演**，没有复制其项目源码。
- 直接依赖：NumPy (BSD), SciPy (BSD), pandas (BSD), matplotlib (PSF-style), PySide6 (LGPL/commercial; **动态链接/打包部署注意授权义务**)；可选 PyTorch (BSD-style)、SimPEG (MIT) 由使用者单独安装。

本项目 LICENSE 仅覆盖本项目自行撰写的代码，不覆盖第三方依赖。二次分发可执行文件时应随包附带第三方许可证与归属信息，检查 PySide6 LGPL 的动态链接和用户替换库义务。

技术备注：本版本的快速成像/异常算法为自编的**晚期半空间近似和阈值型候选区规则**，不是第三方开源项目已经验证的矿井积水 AI 模型。可选 SimPEG 分层地表反演的 API 接口已编写但未在当前构建环境完成依赖运行测试；部署时必须单独进行合成与标定测试。
