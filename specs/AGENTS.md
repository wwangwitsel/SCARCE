# AGENTS.md — SCAR-CLL + DIW 双域实验项目（coding agent 工作守则）

本文件规定 coding agent 的工作方式。技术规格见 `SPEC.md`，分阶段任务与验收标准见 `TASKS.md`。
三份文件冲突时：`SPEC.md` 的数学与数据协议 > `TASKS.md` > 本文件的流程约定。

## 0. 项目一句话

在官方 SCARCE 代码（https://github.com/wwangwitsel/SCARCE ，commit `0d55781`）基础上，实现
**两域都只有 SCAR 互补标签** 的设定下，用 **DIW（每个 mini-batch 求解 KMM）** 估计按互补标签分组的
条件密度比，对训练域 SCARCE 风险加权以逼近测试域风险；交付可复现代码、原始结果与汇总表，支撑一份研究报告。
截止日期 2026-11-27。

## 1. 范围

**必须做**：SPEC.md 中的数据协议、广义 SCAR-OVR 风险、oracle 权重、DIW 权重、5 种方法、关键测试、
结果落盘与汇总脚本、干净环境复现。

**不做**（即使看起来"顺手"也不要做）：ADIW / warm-start PGD 加速、类先验估计（BBE 等）、
support shift / IOS 实验、CIFAR / CLCIFAR、DANN / CLARINET 等额外基线、MLflow、Docker、
超参数大搜索、修改原始 SCARCE 文件的行为。

## 2. 硬性规则

1. **目标域真标签只能出现在 `scarda/evaluate.py` 和 `scarda/oracle.py`（后者仅限训练域池的真标签）**。
   训练、权重估计、模型选择的代码路径不得接触 `y`。不得用目标测试 accuracy 选择 epoch、学习率、B、η 或任何超参数。
2. **不修改根目录的原始 SCARCE 文件**（`main.py`, `utils_algo.py`, `utils_data.py`, `models.py`, `cifar_models/`）。
   新代码全部放在 `scarda/`、`scripts/`、`tests/`、`configs/`。原始 `SCARCE_loss` 作为参照被测试导入。
3. **不擅自改公式**。若认为 SPEC 中某个公式有误，停止实现该部分，写出推导与反例，交给人类确认后再改 SPEC。
4. **检查失败时不要调参让它通过**。风险恒等式、复现测试、数据计数测试失败，一律视为 bug 或规格问题并上报。
5. **不编造结果**。报告中的任何数字必须能在 `results/` 下找到对应文件。没跑过的实验不写成结果。
6. **随机性可追溯**：每次运行记录 seed、split 哈希、config 哈希、git commit 与是否 dirty。
7. 每个里程碑（TASKS.md 的 M0–M6）结束时**停下来汇报**，等待人类确认再进入下一个里程碑。

## 3. 汇报格式（每个里程碑结束时）

- 当前里程碑与每个子任务的状态：已完成 / 进行中 / 待完成
- 证据：执行的命令、关键输出（测试结果、打印的计数、耗时）、产生的文件路径
- 偏离 SPEC 的地方（若有）及原因
- 未解决的问题与风险

不要用"完成了 X%"这类比例描述进度。

## 4. 工程约定

- Python ≥ 3.10，PyTorch 2.x，依赖写入 `requirements.txt`（不沿用原 README 的 Python 3.6 / torch 1.7 版本）。
- 配置用 YAML + dataclass，CLI 只接收 `--config` 与少量覆盖项。
- 所有测试在 CPU 上可跑，`pytest -q` 总耗时 < 90 秒；测试不下载 MNIST（用合成数据）。
- 提交粒度：一个子任务一个 commit，message 以里程碑编号开头，如 `M2: add oracle weights`。
- 结果目录 `results/runs/` 不进 git；`results/summary/` 与 `results/checks/` 进 git。
