# keyData_v4 —— v4 系统全部关键数据台账

> 本文件汇总 financial_agent_system_v4（时间防火墙重构版）的**全部实验数字**及**数据来源文件**，供写论文 / 核对 / 复现。
> 口径：**walk-forward 严格时间隔离**（L1 先验只用 event_date<t / L2 无方向标签 / L3-5 RAG 只检索 pub_date<t）。
> 所有 CSV 路径相对 `financial_agent_system_v4/`。最后更新 2026-07-19。

---

## 0. 定位与缘起

- v4 是 v3（paper_2026 那篇 n=168 论文）之后的**独立重构项目 / 下一篇论文**。paper_2026 不动。
- **缘起**：v3 诊断出校准器 in-sample 泄漏——full-data 61.90% 严格无泄漏后 LOO 掉到 47.62%、OOT 57.58%。
- **v4 目标**：从系统层堵死泄漏（时间防火墙）+ 多专职 skill + 风险防控一级目标。
- **样本**：50 events / 168 场景（沿用 v3 的 car_results.csv），2020-02-20 → 2025-05-20。

---

## 1. 时间防火墙（Phase 0）—— 三条泄漏 + 验证

v3 审计出的三条泄漏，v4 全堵（价格/windowing 通路 v3 已隔离）：

| 泄漏 | 位置 | v4 修复 |
|---|---|---|
| L1 敏感度先验 | 全档案含目标事件 | `firewall.sensitivity_prior(event_dt)` 只用 event_date<t |
| L2 direction 标签 | 生产模式读事件库 | walk-forward 永不读，等价 blind + 断 fallback |
| L3/L4/L5 RAG 零时间过滤 | retrieval 无 event_date；18篇 mechanism 引用2020-24前瞻CAR | `FirewalledRetriever` 只暴露 pub_date<t 文档；18篇前瞻 mechanism 隔离（`mechanism_quarantine.json`），默认语料 `doc_library_events.json`（74篇→事件日过滤） |

- **核心文件**：`firewall.py`（WalkForwardContext + FirewalledRetriever + assert_no_future_leakage）
- **语料拆分**：`scripts/split_doc_library.py` → 92篇 = 74 event-tied + 18 隔离
- **验证**：`test_firewall_logic.py` 全过（过去文档留/未来剔/无日期丢/valid_from放行/源索引不坏）；冒烟实测 `可见文档=45/74`、`先验基于91个历史事件`、时钟锚事件日。

---

## 2. 诚实基线（Phase 0 产物）

**walk-forward 3-seed（base 配置，无任何新 skill）：**

| 指标 | 值 | 来源 |
|---|---|---|
| **方向准确率** | **60.52% ± 0.28pp**（3-seed mean±std）| analyze_multiseed |
| 多数投票 | 61.31% | 同上 |
| 正向 | ~74% | |
| 负向 | 47.29% ± 0.55pp | |
| MAE | 0.0342 | |

- 对比：v3 full-data(泄漏) 61.90% / v3 LOO 47.62%。walk-forward ≈ full-data 但**逐事件 53/168 不同**（firewall 真生效，非复制）。
- **关键洞察**：base 3-seed std 仅 0.28pp → 系统本身高度稳定；之前以为的 ±2.4pp"噪声"主要是 LLM 采样波动。**~60.5% 是无泄漏系统的真实稳定方向上限。**
- 来源 CSV（seed42/123/2024）：`all_experiment_results_qwen2.5_wf_base_20260717_163837.csv` / `..._base_seed123_20260718_105105.csv` / `..._base_seed2024_20260718_212215.csv`

---

## 3. Phase 1 — 标量→传导张量契约（行为保持）

- `contracts.py` `TransmissionMap`：`raw_factors`（entity键，逐位复刻 v3 `to_exogenous_factors`：`+`→+1其余−1、_MAG_MAP、last-write-wins）驱动标量；`cells`（申万码聚合，fine_dir ±1/0）供 Phase 3/4。
- `as_exog_factors/project_to_scalar/event_strength` 全走 raw_factors → 数字与 Phase 0 完全一致。
- 验证：`test_firewall_logic.py` 边角 case（重复entity/未知方向/空方向）断言全过；端到端平价冒烟数值链同构。
- `result["transmission"]` 张量视图供下游消费。

## 4. Phase 3 — 传导 skill + DSGE IRF 先验（开关 USE_IRF）

- **IRF 先验 `data/irf_priors.json`**：47 个 (event_type,industry) cell，方向来自传导经济学、经 workflow 对抗验证（run wf_70aa3f53-f59）。方向分布：market_event 全−7 / monetary +5 / geopolitical +2−9 / industry +11−2/0:2 / capital_market +8/0:1。强度按 n<5 收缩。构建：`dump_irf_skeleton.py`→`build_irf_priors.py`。
- **软混合**：`transmission_agent.py`，`llm_sig=tanh(net/2)`，`blend=(1-λ)llm_sig+λ·irf_sig`，λ=0.35 预注册固定。
- **firewall 合规**：IRF 方向用全局理论值（无泄漏），强度按 <t 历史 P_pos 一致性 `2|P_pos-0.5|` 缩放，n<3 再 ×0.3，冷启动抽身。
- 单测 `test_transmission.py` 过（弱先验不翻转强LLM/强冲突拉偏/未知透传）。

## 5. Phase 2 — 基本面 skill

### part1 价格派生（开关 USE_FUNDAMENTALS）
- `fundamentals_agent.py` `FundamentalsSkill.probe`：复用 `slice_history_before_event`（严格<t）算市场层(rel_strength/vol_regime)、行业层(momentum/valuation_pos)、流动层(volume_trend)。写 cell path/confidence，不改方向。
- 隔离验证：probe(全量)==probe(截断<t) 逐位一致。

### part3-A 真实行业估值（开关 USE_FUND_VALUATION）
- 数据 `scripts/fetch_industry_fundamentals.py`（akshare `index_analysis_daily_sw(symbol='一级行业')`，分年+缓存+重试）→ `data/raw/industry_fundamentals.csv`：**20行业×1817行 PE/PB/股息率，2019-01→2026-07**。
- `probe_valuation`：<t 严格切算 PE/PB 历史分位+PE动量+股息率。`valuation_signal=-(pe_pctile-0.5)*2`（便宜→+/贵→−）。
- 隔离验证过（full==truncated）。样例：银行2024-07-22 PE=5.3 pctile=35% div=5.45% signal=+0.304。
- **part3-B（个股财报）不做**：A 直接估值面已证伪方向增量，B 大概率同样无效且代价大（成分股时变泄漏+公告日bug）。

---

## 6. Phase 4 — 风险防控（独立一级目标）

- `evaluation/risk_metrics.py`：`RiskHead`（event_VaR=conformal区间下界 / expected_MDD / tail_prob=α(1+dispersion)）+ `backtest_var`（Kupiec POF / Christoffersen 独立性 / pinball，**自实现卡方p值无scipy**，分布无关不EVT）。消费 `transmission.dispersion()`。
- 单测 `test_risk.py` 过（Kupiec 校准5/100不拒/过度20/100拒；Christoffersen 聚簇LR≫分散）。

---

## 7. 消融矩阵 + 3-seed 聚合（核心结果表）

**5 配置单 seed（seed42，`run_ablation_matrix.py`）：**
| config | overall | pos | neg | MAE |
|---|---|---|---|---|
| base | 60.71% | 74.39% | 47.67% | 0.0345 |
| irf | 61.90% | 75.61% | 48.84% | **0.0252** |
| fund | 60.12% | 75.61% | 45.35% | 0.0339 |
| fundval | 60.12% | 73.17% | 47.67% | 0.0346 |
| all | 60.71% | 73.17% | 48.84% | 0.0255 |

**base/irf/all 3-seed 聚合（`run_multiseed.py`+`analyze_multiseed.py`，seed 42/123/2024）：**
| config | overall mean±std | neg mean±std | MAE | maj-vote |
|---|---|---|---|---|
| base | 60.52±0.28pp | 47.29±0.55 | 0.0342 | 61.31% |
| irf | 60.52±1.01pp | 48.06±1.10 | **0.0252** | 60.71% |
| all | 59.33±1.01pp | 46.51±1.64 | 0.0258 | 60.12% |

**显著性（paired bootstrap，`analyze_ablation_matrix.py`）**：所有配置方向差异 p>0.05（irf overall p=0.462 / fund 0.826 / fundval 0.794 neg Δ=+0.00 p=1.000 / all 1.000）。**方向增量全部不显著=噪声。**

## 8. VaR 回测矩阵（`analyze_risk_matrix.py`，base/irf/all × 3seed）

| config | breach% | Kupiec p | Christ p | pinball | MDD MAE |
|---|---|---|---|---|---|
| base | 0.6% | 0.001(FAIL) | 0.913 | 0.0114 | 0.0784 |
| irf | 0.6% | 0.001(FAIL) | 0.913 | 0.0113 | **0.0546** |
| all | 0.6% | 0.001 | 0.913 | 0.0114 | 0.0554 |

- VaR 偏保守（breach 0.6% vs α=5%）=风险管理安全侧失败（conformal 区间下界天然偏宽）。
- Christoffersen p=0.913 通过：违约不聚簇。

## 9. 证据总账（全部跨 3-seed）

- **IRF 正贡献（3 独立证据）**：点预测 MAE −26%（0.0342→0.0252）/ 回撤 MDD MAE −30%（0.078→0.055）/ VaR 保守但违约独立。**方向 +0.00pp（无效）。**
- **方向负结果**：三类轻量语义信号（理论先验/价格动量/真实估值）方向增量全=0，**60.5%±0.28 是无泄漏稳定上限**。
- **风险目标独立成立**：即使方向到顶，VaR 可回测、违约独立、IRF 改善回撤刻画。

## 10. 论文四大论点（写作骨架）

1. **方法学**：量化+消除 LLM 事件研究 in-sample 泄漏（61.90%→无泄漏 60.5%），可复用防火墙协议。
2. **正结果**：DSGE-IRF 传导先验跨-seed 稳健改善幅度/风险刻画（MAE/MDD/VaR），不改方向。
3. **风险一级目标**：事件级 VaR 独立于方向评测（Kupiec/Christoffersen/pinball）。
4. **诚实负结果**：轻量语义信号纠不动方向，~60.5% 是稳定上限，为社区标定边界 + 指明突破口（更强监督/大样本/事件级 catalyst）。

## 11. 脚本清单（结论一键复现）

| 用途 | 脚本 |
|---|---|
| 单配置 walk-forward | `scripts/run_walk_forward.py`（WALK_FORWARD/USE_IRF/USE_FUNDAMENTALS/USE_FUND_VALUATION/RUN_TAG/SEED 环境变量） |
| 消融矩阵（子进程隔离） | `scripts/run_ablation_matrix.py [--only ...] [--seed N]` |
| 多 seed 降噪 | `scripts/run_multiseed.py` |
| 方向显著性 | `scripts/analyze_ablation_matrix.py --nboot 5000` |
| 3-seed 聚合 | `scripts/analyze_multiseed.py` |
| VaR 回测（单/批） | `scripts/analyze_risk.py` / `scripts/analyze_risk_matrix.py` |
| 数据准备 | `scripts/split_doc_library.py` / `fetch_industry_fundamentals.py` / `dump_irf_skeleton.py` / `build_irf_priors.py` |
| 单测 | `test_firewall_logic.py` / `test_transmission.py` / `test_risk.py` / `test_fundamentals.py` |

## 12. 四个正交消融开关

`ABLATION_MODE`(A1/A2/A3) / `WALK_FORWARD` / `USE_IRF` / `USE_FUNDAMENTALS` / `USE_FUND_VALUATION`。**正式实验一律走 `run_ablation_matrix.py`（子进程隔离，免疫 $env 残留）；手动 $env 仅冒烟且命令前必清残留。**
