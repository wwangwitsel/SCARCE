# SPEC.md — 技术规格（v1，冻结前草案）

## 1. 已核对的 SCARCE 源码事实（commit `0d55781`）

以下事实来自实际阅读源码，实现时必须据此行事：

| 位置 | 事实 | 对本项目的含义 |
|---|---|---|
| `utils_algo.py` L12–42 `SCARCE_loss` | 输入 `labels` 是**单个**互补标签的类别索引，内部转 one-hot（L14–15） | 需要推广为多热矩阵 $\bar Y\in\{0,1\}^{n\times K}$ |
| `utils_algo.py` L34–36 | 类先验**写死为均匀** $1/K$；变量名 `ccp` 在函数内部表示 $1-\pi_k$ | 测试域先验非均匀，必须显式传入 $\pi$ |
| `utils_algo.py` L56 `chosen_loss_c` | 接收参数 `ccp`（来自 `utils_data.py` L87，互补标签频率），但**未传给** `SCARCE_loss`，被忽略 | 不要沿用这个参数链路 |
| `utils_algo.py` L25–29 | "unlabeled" 集合 = batch 内**全部**样本 | 与第 3 节 A/B 形式在 $\bar\pi$=batch 频率时数值等价（已用 numpy 移植验证，差 ~1e-15） |
| `utils_algo.py` L38–40 | 风险修正为逐类 **`torch.abs`**（ABS 修正），作用于正类部分风险 | 默认沿用 `abs`；报告中写明"沿用官方代码的 ABS 修正" |
| `utils_algo.py` L22–23 | 某类在 batch 中无互补标签样本时，分母置 1，该项为 0 | 空组的处理沿用此约定，并计数 |
| `utils_data.py` L34–45 `random` | 每个样本恰好一个互补标签，在 $K-1$ 个非真类中均匀抽取 → 满足 SCAR，$c_k=1/(K-1)$ | 可作为一种生成模式 |
| `utils_data.py` `set1`/`set2` | 转移矩阵依赖真类 → **不满足 SCAR** | 不使用 |
| `main.py` L106–… | 报告指标 = 最后 10 个 epoch 的测试 accuracy 均值，无模型选择 | 沿用为主指标（固定规则，不涉及选择） |
| `main.py` L97, 记录行 | 日志中的 train_loss 是最后一个 batch 的值 | 新代码记录 epoch 平均 |
| `models.py` `mlp_model` | 784→500→ReLU→K | 主模型；需要能取出 500 维隐层作为 DIW 表示 |

## 2. 设定与记号

- 类别 $\mathcal Y=\{0,\dots,K-1\}$，$K=10$；域 $d\in\{\mathrm{tr},\mathrm{te}\}$。
- 观测：$(x,\bar{\mathbf y})$，$\bar{\mathbf y}\in\{0,1\}^K$，$\bar y_k=1$ 表示 $k$ 是互补标签。允许全零行（无标签样本）。
- 类先验 $\pi_{d,k}=P_d(Y=k)$；互补标签率 $\bar\pi_{d,k}=P_d(\bar Y_k=1)$。
- 每域 SCAR：$P_d(\bar Y_k=1\mid x,Y\neq k)=c_{d,k}$，$P_d(\bar Y_k=1\mid x,Y=k)=0$，故 $\bar\pi_{d,k}=c_{d,k}(1-\pi_{d,k})$。允许 $c_{\mathrm{tr},k}\neq c_{\mathrm{te},k}$。
- 损失 $\ell(z)=\log(1+e^{-z})$（与 `logistic_loss` 一致）；$f_k$ 为第 $k$ 个输出。
- 条件密度比 $r_k^{+}(x)=\dfrac{p_{\mathrm{te}}(x\mid\bar Y_k=1)}{p_{\mathrm{tr}}(x\mid\bar Y_k=1)}$，$r_k^{-}(x)=\dfrac{p_{\mathrm{te}}(x\mid\bar Y_k=0)}{p_{\mathrm{tr}}(x\mid\bar Y_k=0)}$。假设对应支撑条件成立。

## 3. 风险（核心公式，不得擅改）

测试域 OVR 风险 $R_{\mathrm{te}}(F)=\mathbb E_{\mathrm{te}}\big[\ell(f_y)+\sum_{k\neq y}\ell(-f_k)\big]=\sum_k R_{\mathrm{te},k}$，
在测试域 SCAR 下：

$$R_{\mathrm{te},k}=\underbrace{(1-\pi_{\mathrm{te},k})\,\mathbb E_{p_{\mathrm{te}}(x\mid\bar Y_k=1)}[\ell(-f_k)]}_{N_k}+\underbrace{(\bar\pi_{\mathrm{te},k}+\pi_{\mathrm{te},k}-1)\,\mathbb E_{p_{\mathrm{te}}(x\mid\bar Y_k=1)}[\ell(f_k)]+(1-\bar\pi_{\mathrm{te},k})\,\mathbb E_{p_{\mathrm{te}}(x\mid\bar Y_k=0)}[\ell(f_k)]}_{P_k\ (=\pi_{\mathrm{te},k}\mathbb E_{\mathrm{te}}[\ell(f_k)\mid Y=k]\ \ge 0)}$$

用训练域样本表示：把测试域条件期望替换为训练域对应组内以 $r_k^{\pm}$ 加权的期望。

### 3.1 统一的经验风险函数

```
scar_ovr_risk(logits[n,K], ybar[n,K], pi[K], pibar[K],
              w_pos[n,K] | None, w_neg[n,K] | None,
              correction in {"abs","relu","none"}) -> RiskOut
```

- 组内（自归一化）加权均值：$M^{+}_k(g)=\dfrac{\sum_{i:\bar y_{ik}=1} w^{+}_{ik}g_{ik}}{\sum_{i:\bar y_{ik}=1} w^{+}_{ik}}$，$M^{-}_k$ 同理用 $\bar y_{ik}=0$ 和 $w^-$。`None` 视为全 1。空组 → 该均值取 0 并计数（与 SCARCE L22–23 一致）。
- $\hat N_k=(1-\pi_k)M^+_k(\ell(-f_k))$
- $\hat P_k=(\bar\pi_k+\pi_k-1)M^+_k(\ell(f_k))+(1-\bar\pi_k)M^-_k(\ell(f_k))$
- $\hat R=\sum_k\big[\hat N_k+\mathrm{corr}(\hat P_k)\big]$，`abs`→$|\cdot|$，`relu`→$\max(0,\cdot)$，`none`→恒等。
- `RiskOut` 返回：`total`, `N[K]`, `P_raw[K]`, `neg_rate`（$\hat P_k<0$ 的比例）, `empty_groups`。
- 自归一化使估计量为比值估计，有限样本有偏；报告中必须如此描述。

### 3.2 $\bar\pi$ 的取值模式

- `pibar_mode=batch`：用当前 batch 的 $\bar y$ 列均值。仅用于**无权重**方法，以与官方 SCARCE 数值一致。
- `pibar_mode=fixed`：用传入的固定向量。加权方法必须用 `fixed`，取 $\hat{\bar\pi}_{\mathrm{te}}$ = 测试域训练池 `T_tr` 的经验频率（可观测量）。

### 3.3 各方法的目标函数

| 方法 id | 数据 | $\pi$ | $\bar\pi$ | 权重 | 备注 |
|---|---|---|---|---|---|
| `source_only` | S_tr | $\pi_{\mathrm{tr}}$ | batch | 无 | 官方 SCARCE 的直接推广 |
| `target_only` | T_tr | $\pi_{\mathrm{te}}$ | batch | 无 | 小样本 |
| `oracle_iw` | S_tr | $\pi_{\mathrm{te}}$ | fixed $\hat{\bar\pi}_{\mathrm{te}}$ | oracle（第 5 节） | 参照，使用特权信息 |
| `diw` | S_tr（+ T_tr 仅用于估权重） | $\pi_{\mathrm{te}}$ | fixed $\hat{\bar\pi}_{\mathrm{te}}$ | DIW（第 6 节） | η=0 |
| `diw_combined` | S_tr + T_tr | $\pi_{\mathrm{te}}$ | 同上 | DIW | $J=\eta\hat R^{\mathrm{T\_tr}}+(1-\eta)\hat R^{\mathrm{DIW}}$，$\eta=n_{te}/(n_{tr}+n_{te})$，两项各自修正 |

$\pi_{\mathrm{te}}$ 在 v1 中**使用已知真值**（来自配置），报告中声明为 oracle prior。类先验估计不在范围内。

## 4. 数据协议

### 4.1 数据源与池

- 训练基数据：torchvision MNIST train（60k）；评估基数据：MNIST test（10k）。预处理只用 `ToTensor()`（与 SCARCE 一致）。
- 由 `split_seed` 在每个类内部打乱索引，按类计数切出**互不相交**的池：

| 池 | 来源 | 大小 | 先验 | 可见信息 |
|---|---|---|---|---|
| `S_tr` | MNIST train | $n_{tr}=20000$ | $\pi_{\mathrm{tr}}$ = 均匀 | $x,\bar y$（$c_{tr}$） |
| `T_tr` | MNIST train | $n_{te}\in\{500,2000\}$ | $\pi_{\mathrm{te}}$ | $x,\bar y$（$c_{te}$） |
| `T_val` | MNIST train | 2000 | $\pi_{\mathrm{te}}$ | $x,\bar y$（$c_{te}$），仅用于模型选择 |
| `T_eval` | MNIST test | 3000 | $\pi_{\mathrm{te}}$ | $x,y$，仅用于最终评估 |

- 类计数：$n\cdot\pi$ 用最大余数法取整，**精确等于**该计数（不是伯努利抽样）。可用量不足时报错。已核算：$\rho=20$ 时 `T_eval` 最大可行规模约 3337，训练侧每类余量 > 2700。
- `T_tr` 的 $n_{te}=500$ 版本是 2000 版本按类取前缀得到的子集（嵌套，便于配对比较）。

### 4.2 偏移

- $\pi_{\mathrm{tr}}$ 均匀；$\pi_{\mathrm{te},k}\propto\rho^{-k/(K-1)}$，$\rho\in\{5,20\}$；另设 $\rho=1$ 作为无偏移对照（仅少量运行）。
- 这是**类先验偏移**：$p_{\mathrm{te}}(x\mid y)=p_{\mathrm{tr}}(x\mid y)$。选择它是因为 oracle 权重可精确计算。报告须声明仅覆盖此偏移类型。

### 4.3 互补标签生成

- 默认 `gen=bernoulli`：对每个样本、每个 $k\neq y$，独立以概率 $c_d$ 置 $\bar y_k=1$。默认 $c_{tr}=0.3$，$c_{te}=0.2$（所有 $k$ 相同）。
- 可选 `gen=single_uniform`：等同 SCARCE `random`（$c=1/(K-1)$）。
- 由 `label_seed` 控制；每个池独立生成。

### 4.4 可见性隔离（必须由代码结构保证）

- 训练侧只拿到 `WeakSet(x, ybar)`；真标签放在单独的 `LabelVault`，仅 `evaluate.py` 与 `oracle.py` 可读。
- `oracle.py` 只读取 `S_tr` 的真标签。
- 记录 split 哈希：各池索引数组与 $\bar y$ 数组的 sha256。

### 4.5 Seed

一个 `seed` 决定 split、互补标签与模型初始化（`split_seed=label_seed=init_seed=seed`）。同一 seed 下所有方法共享同一份数据，形成配对比较。主实验 seeds = {0,1,2}。

## 5. Oracle 权重（类先验偏移下的精确形式）

对 `S_tr` 中真标签为 $y$ 的样本 $i$，在组 $(k,s)$ 中：

- $s=+$（$\bar y_{ik}=1$，故 $y\neq k$）：$w^{+}_{k}(y)=\dfrac{\pi_{\mathrm{te},y}/(1-\pi_{\mathrm{te},k})}{\pi_{\mathrm{tr},y}/(1-\pi_{\mathrm{tr},k})}$（$c$ 在组内相消）
- $s=-$（$\bar y_{ik}=0$）：$w^{-}_{k}(y)=\dfrac{\pi_{\mathrm{te},y}\,(1-c_{\mathrm{te}}\mathbb 1[y\neq k])/(1-\bar\pi_{\mathrm{te},k})}{\pi_{\mathrm{tr},y}\,(1-c_{\mathrm{tr}}\mathbb 1[y\neq k])/(1-\bar\pi_{\mathrm{tr},k})}$，其中 $\bar\pi_{d,k}=c_d(1-\pi_{d,k})$ 用**总体值**

依据：$p(x\mid y)$ 跨域不变，故对任意只依赖 $x$ 的 $g$，$\mathbb E_{\mathrm{tr}}[w(y)g(x)\mid\bar Y_k=s]=\mathbb E_{\mathrm{te}}[g(x)\mid\bar Y_k=s]$。总体下组内权重均值为 1。精确值不再按样本均值归一化（但 3.1 的自归一化仍会作用，报告中说明）。

## 6. DIW 权重估计

每个训练 step：

1. 取训练 batch $B_{tr}$（S_tr，bs=256）和测试 batch $B_{te}$（从 T_tr 有放回抽取，大小 $\min(256,n_{te})$）。
2. 表示 $z=\phi_\theta(x)$：MLP 的 500 维 ReLU 隐层，`torch.no_grad()` 计算并 detach；权重不回传梯度。
3. RBF 核带宽 $\sigma$ = 本 step 中 $B_{tr}\cup B_{te}$ 的 $z$ 两两距离中位数。
4. 对每个 $(k,s)$：$G_{tr}=\{i\in B_{tr}:\bar y_{ik}=s\}$，$G_{te}=\{j\in B_{te}:\bar y_{jk}=s\}$。若任一组小于 `min_group`（默认 4），该组权重置 1 并计入 `fallback`。
5. KMM：$m=|G_{tr}|$，$m'=|G_{te}|$，
   $\min_w \tfrac12 w^\top K w-\kappa^\top w$，$K_{ij}=k(z_i,z_j)$，$\kappa_i=\tfrac{m}{m'}\sum_{j\in G_{te}}k(z_i,z_j)$，
   约束 $0\le w_i\le B$，$|\sum_i w_i-m|\le m\,\epsilon$。默认 $B=10$，$\epsilon=0.05$。
6. 求解器：投影梯度法，**每个 step 从 $w=\mathbf 1$ 重新开始、求解至收敛**（相对目标变化 < 1e-6 或 1000 次迭代），步长 $1/\lambda_{\max}(K)$。投影到 {盒约束 ∩ 和约束带}：对标量平移 $\lambda$ 二分，$w=\mathrm{clip}(v-\lambda,0,B)$。允许把 $2K$ 个组 padding 后批量求解。
   - 不使用跨 step 的 warm start、不限制为少量迭代（那是 ADIW 的做法，不在本项目范围）。
   - 报告中描述为"DIW 式的逐 mini-batch KMM，以投影梯度法求解至收敛"。
7. 得到 $w^+_{ik}$（$i\in G_{tr}(k,+)$）与 $w^-_{ik}$（$i\in G_{tr}(k,-)$），送入 3.1。
8. Warm-up：前 `diw_warmup_epochs`（默认 5）个 epoch 使用 `source_only` 目标，之后切换。记录切换点。

## 7. 模型、训练与选择

- 模型：根目录 `models.mlp_model(784, 500, 10)`；在 `scarda/models.py` 中子类化，增加 `features(x)` 返回隐层，`forward` 不变。
- 优化：Adam，lr=1e-3，wd=1e-5，bs=256（SCARCE README 默认值）。
- 训练预算按**优化步数**统一：每"epoch" = $\lceil n_{tr}/bs\rceil$ 步，所有方法（含 `target_only`，其 loader 循环使用）相同。epoch 数 `E` 在 M1 测完耗时后冻结，草案默认 100。
- 每 epoch 记录 `T_val` 上的弱标签验证风险：`scar_ovr_risk`，`correction=none`，$\pi_{\mathrm{te}}$，$\bar\pi$ = T_val 经验频率，无权重。
- 模型选择：**仅对 DIW 的 $B\in\{5,10\}$ 做选择**，规则 = 在 seed 0、$\rho=20$、$n_{te}=2000$ 上，最后 10 个 epoch 的平均 `T_val` 弱标签风险最小者。选定后冻结并写入 `configs/frozen.yaml`。其余超参数固定不调。
- 主指标：`T_eval` accuracy 的最后 10 个 epoch 均值（沿用 SCARCE）；同时报告最后一个 epoch 的 accuracy 与 balanced accuracy。

## 8. 输出格式

`results/runs/{method}_rho{ρ}_nte{n}_seed{s}_{cfghash8}/`：

- `config.yaml`：完整解析后的配置
- `meta.json`：git commit、dirty 标志、python/torch 版本、device、split 哈希、开始/结束时间、耗时
- `metrics.csv`：每 epoch 一行：`epoch, step, train_loss_mean, val_weak_risk, target_acc, target_bal_acc, neg_rate, empty_groups, ess_pos_mean, ess_neg_mean, w_max, fallback, kmm_iters_mean, epoch_time`
- `final.json`：`acc_last10, acc_last, bal_acc_last10, bal_acc_last`

ESS 比例定义为 $(\sum w)^2/(m\sum w^2)$，按组计算后对 $(k,s)$ 取平均。`final.json` 存在时跳过该 run（支持断点续跑）。

汇总：`scripts/summarize.py` 只读 `results/runs/`，输出 `results/summary/main_table.{csv,md}`、`paired_diffs.csv`（各方法减 `source_only`，按 seed 配对，给出均值、标准差、95% t 区间）、`fig_acc_vs_nte.png`、`fig_ess.png`，并生成 `results/summary/REPORT_FACTS.md`（报告可引用的数字清单，每条注明来源文件）。

## 9. 主实验矩阵（M3 冻结时确认）

方法 {source_only, target_only, oracle_iw, diw, diw_combined} × $\rho$ {5, 20} × $n_{te}$ {500, 2000} × seed {0,1,2}。
`source_only` 与 $n_{te}$ 无关，去重后共 54 次。另加 $\rho=1$ 下 `source_only` 与 `diw` 各 1 个 seed 作为无偏移对照。若 M1 测得耗时不允许，优先砍 $\rho=5$，不砍 seed。

## 10. 目录结构

```
SCARCE/                      # fork，根目录原始文件保持不变
  scarda/  __init__.py config.py data.py risk.py oracle.py diw.py models.py
           train.py evaluate.py io_utils.py
  scripts/ run.py run_matrix.py summarize.py check_identity.py bench_time.py
  configs/ base.yaml smoke.yaml matrix_main.yaml frozen.yaml
  tests/   test_risk.py test_data.py test_oracle.py test_diw.py test_isolation.py test_smoke.py
  results/ runs/(gitignore) checks/ summary/
  requirements.txt  .github/workflows/ci.yml  README_SCARDA.md
```

入口：`python scripts/run.py --config configs/base.yaml method=diw rho=20 n_te=2000 seed=0`
