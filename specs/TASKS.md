# TASKS.md — 里程碑与验收标准

每个里程碑结束后停下来按 AGENTS.md §3 汇报，等人类确认再继续。日期为计划日期，不代表已完成。

---

## M0 环境与基线复现（11/20）

1. Fork 仓库，新建 `requirements.txt`（Python ≥3.10、torch 2.x、torchvision、numpy、scipy、pyyaml、pandas、matplotlib、pytest）。
2. 在新环境下运行原始 `main.py` 的缩短版（`-ep 2 -run_times 1`），确认原代码可跑。
3. 实现 `scarda/risk.py::scar_ovr_risk`（SPEC §3.1–3.2）。
4. `tests/test_risk.py`：
   - **T1 复现**：单热 $\bar y$、均匀 $\pi$、`pibar_mode=batch`、无权重、`abs` → 与根目录 `utils_algo.SCARCE_loss` 相差 < 1e-6（多组随机 logits，含某类空组的情形）。
   - **T2 单位权重**：`w_pos=w_neg=1` 与 `None` 结果相同。
   - **T3 修正分支**：构造 $\hat P_k<0$ 的输入，`abs`/`relu`/`none` 三种结果符合定义，`neg_rate` 正确。

**验收**：原 `main.py` 跑通的日志；`pytest tests/test_risk.py` 全部通过。

---

## M1 数据协议 + source_only 端到端（11/20–21）

1. `scarda/data.py`：按 SPEC §4 构造四个池、偏移先验、互补标签、`WeakSet` / `LabelVault`、split 哈希。
2. `scarda/models.py`、`scarda/train.py`（先只支持 `source_only`、`target_only`）、`scarda/evaluate.py`、`scarda/io_utils.py`、`scripts/run.py`。
3. `tests/test_data.py`（**T4**）：各池类计数精确等于目标计数；池两两不相交；$\bar y$ 从不包含真类；经验 $c$ 与设定值的偏差在 3 个标准误内；同 seed 两次生成哈希一致。
4. `scripts/bench_time.py`：测 `source_only` 每 epoch 耗时（以及 GPU/CPU 信息）。
5. 在真实 MNIST 上跑 `source_only`，$\rho=20$，seed 0，完整 epoch 数。

**验收**：`results/runs/` 下有一个完整 run 的 4 个文件；打印出的各池实际先验与 $\hat{\bar\pi}$ 与设定一致；给出单 epoch 耗时，据此提出 `E` 的冻结值与矩阵规模建议（不自行决定）。

---

## M2 Oracle 权重 + 风险恒等式检查 + target_only（11/21）

1. `scarda/oracle.py`（SPEC §5），`train.py` 支持 `oracle_iw` 与 `target_only`。
2. `tests/test_oracle.py`（**T5**）：在合成 3 类高斯混合 + 类先验偏移数据上，大样本下每个 $(k,s)$ 组 oracle 权重均值 ≈ 1。
3. `tests/test_risk.py` 增加 **T6 恒等式（合成数据）**：固定线性分类器，`correction=none`，以下三者在重复抽样均值上相差不超过 3 个标准误：
   (a) 带真标签的测试域 OVR 风险；(b) 测试域弱标签 SCAR 风险（真 $\pi_{te}$）；(c) oracle 加权训练域风险。
4. `scripts/check_identity.py`：在 MNIST 上对两个固定分类器（随机初始化、source_only 训练 1 epoch 后）计算同样三者，输出到 `results/checks/identity.json`。
5. 跑 `oracle_iw`、`target_only` 各 1 个 run（$\rho=20$，$n_{te}=2000$，seed 0）。

**验收**：T5、T6 通过；`identity.json` 中三者一致（差异在给出的标准误范围内）；两个 run 完整落盘。
**止损**：恒等式不一致 → 停止，不进入 M3，上报推导与数值。若 oracle_iw 的 accuracy 不优于 source_only，不视为失败，照常记录。

---

## M3 DIW + 冻结（11/22）

1. `scarda/diw.py`（SPEC §6），`train.py` 支持 `diw`、`diw_combined`，记录全部诊断字段。
2. `tests/test_diw.py`（**T7**）：
   - 求解器与 `scipy.optimize.minimize`（SLSQP 或 trust-constr）在随机小问题上目标值相对差 < 1e-4，且约束满足；
   - $G_{te}$ 与 $G_{tr}$ 是同一组样本时，权重 ≈ 1；
   - 空组 / 小组触发 fallback 且计数正确。
3. `tests/test_isolation.py`（**T8**）：训练入口的签名与数据对象不含真标签；`T_eval` 索引不出现在任何训练侧对象中；用 `ast`/grep 检查训练与权重模块未导入 `LabelVault`。
4. `tests/test_smoke.py`（**T9**）：合成数据、2 个 epoch，5 种方法各自跑通并写出全部输出文件。
5. 按 SPEC §7 对 $B\in\{5,10\}$ 做选择，写入 `configs/frozen.yaml`；测 DIW 单 epoch 耗时。
6. `.github/workflows/ci.yml`：CPU 上运行 `pytest -q`。

**验收**：全部测试通过；`frozen.yaml` 提交；人类确认冻结方法、数据协议、选择规则与矩阵（SPEC §9）后，M3 结束。冻结后只允许修 bug，不加功能。
**止损**：若 DIW 训练出现 NaN / 权重退化（ESS 比例长期 < 0.05）且当天无法定位，上报；主结果改为 source/target/oracle 三者，DIW 作为部分结果与失败分析。

---

## M4 主实验（11/23–24）

1. `scripts/run_matrix.py` 读取 `configs/matrix_main.yaml`，支持断点续跑与并行度参数。
2. 跑完整矩阵 + $\rho=1$ 对照。

**验收**：每个 run 都有 `final.json`；失败的 run 有日志说明原因；不得因结果不理想而删除或重跑某个 seed。

---

## M5 汇总与分析材料（11/25）

1. `scripts/summarize.py`（SPEC §8 输出）。
2. 生成 `REPORT_FACTS.md`：每个方法 × 条件的均值 ± 标准差、配对差及区间、ESS 与 fallback 统计、修正触发率、总运行次数与总耗时，每条注明来源文件。

**验收**：删除 `results/summary/` 后重跑 `summarize.py` 能得到完全相同的文件。

---

## M6 复现验收（11/26–27）

1. 写 `README_SCARDA.md`：设定、与原 SCARCE 的差异、安装、smoke 命令、单个真实实验命令、矩阵命令、汇总命令、已知局限（oracle prior、仅类先验偏移、单数据集、自归一化有偏、ABS 修正沿用官方代码）。
2. 在全新环境中全新 clone：安装 → `pytest -q` → smoke → 一个真实 run → `summarize.py`。

**验收**：上述步骤的完整终端记录保存为 `results/checks/clean_env_log.txt`。
