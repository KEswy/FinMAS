# FinMAS 简历回填 Markdown

> 项目名称：FinMAS: An LLM-driven Multi-Agent Financial Analysis System
>
> 中文名称：基于 LLM 的多智能体金融分析系统
>
> 风控中心模块：FinMAS-Risk

---

## 一、项目一句话

FinMAS 是一个面向中国 A 股宏观与政策事件的 LLM 多智能体金融分析系统，融合机理推理、RAG、时序预测与风险归因，在严格 walk-forward 时间防火墙下完成事件收益预测和风险监控。

---

## 二、系统整体架构

### 2.1 FinMAS 主系统

```text
输入：
  事件日期
  事件描述
  受影响行业代码

核心链路：
  Mechanism Agent（机理 Agent）
      ↓
  Transmission Map（跨层传导图）
      ↓
  Data Agent（时序预测）
      ↓
  外生信号融合与方向修正
      ↓
  统一评估与风险头计算
```

主要模块：

| 模块 | 作用 |
|---|---|
| Mechanism Agent | 基于 RAT × KG 生成“社会 → 行业 → 公司 → 资产”事件机理链 |
| RAG | BGE 中文向量 + FAISS，严格过滤事件日之后的文档 |
| Temporal Firewall | L1/L2/RAG 三层时间防火墙，禁止未来标签和数据泄漏 |
| Data Agent | Informer + KF-Transformer + GatedFusion 双路时序预测 |
| Conformal Prediction | 生成 80%/95% 预测区间 |
| Risk Head | 输出事件级 VaR、MDD、尾概率、风险区间 |
| LLM Provider | 支持 DeepSeek V4 Flash API 和本地 Ollama，支持并发调用 |

### 2.2 FinMAS-Risk 风控中心模块

> `【FinMAS-Risk / 风控中心实习】`

FinMAS-Risk 是在实验系统之上新增的离线风控分析服务，不重新训练模型，直接复用 321 场景实验结果与原始市场数据。

```text
FinMAS-Risk
  ├── data_loader.py      原始数据与实验结果加载
  ├── risk_models.py      历史模拟 / Bootstrap / 参数法 / EWMA / EVT / 模型区间
  ├── portfolio_risk.py   事件级、行业级、组合级风险聚合与 VaR 回测
  ├── stress_test.py      历史情景 + 合成市场冲击
  ├── attribution.py      收益归因 + Component VaR / Marginal VaR
  ├── llm_commentary.py   DeepSeek 风险点评
  ├── report.py           JSON / CSV / Plotly HTML 报告
  ├── api.py              FastAPI 接口
  └── cli.py              命令行入口
```

---

## 三、适合写进简历的项目描述

### 3.1 总体项目描述

```text
FinMAS：基于 LLM 的多智能体金融分析系统

面向中国 A 股宏观政策事件，构建“LLM 多智能体 + RAG + 时序预测 + 风险归因”的端到端分析系统。
系统采用 walk-forward 时间防火墙，严格避免未来数据泄漏，并在 321 个事件-行业场景上完成 GPU 全量回测。
```

### 3.2 主系统工作内容

```text
- 设计并实现 Mechanism Agent，将宏观事件转化为多层机理链，作为外生因子注入时序预测。
- 构建基于 BGE 向量与 FAISS 的事件级 RAG 检索器，实现事件日前可见文档的时间防火墙。
- 实现 Informer + KF-Transformer 双路时序预测，结合 Conformal Prediction 输出预测区间。
- 接入 DeepSeek V4 Flash API 与本地 Ollama，支持多模型切换和并发实验。
- 在 RTX 5090 GPU 上完成 321 个事件-行业场景的 walk-forward 全量回测。
```

### 3.3 FinMAS-Risk 风控中心实习重点

> `【FinMAS-Risk / 风控中心实习】`

```text
- 负责完善金融事件分析系统的风险评估模块，新增 FinMAS-Risk 离线风控服务。
- 实现事件级、行业级、组合级三级风险视图，覆盖 T+1/T+5/T+20 多期限。
- 实现可切换风险模型：历史模拟、Bootstrap、参数法、EWMA、EVT、模型区间。
- 搭建 VaR 回测体系：Kupiec POF、Christoffersen 独立性、Pinball Loss、Bootstrap CI。
- 构建压力测试模块，支持历史情景重放与合成市场冲击。
- 实现收益归因和风险归因，包括 60 日滚动 beta、市场/行业/事件 alpha 分解、Component VaR。
- 输出可审计 JSON/CSV 报告和 Plotly HTML 风险仪表盘。
- 通过 FastAPI 提供风险报告、压力测试、归因报告的读取与提交接口。
- 使用 DeepSeek 对聚合风险结果生成自然语言风控点评。
```

---

## 四、关键量化结果

### 4.1 实验规模

```text
321 个事件-行业场景
96 个事件日期
18 个申万一级行业
3 个风险期限：T+1 / T+5 / T+20
GPU：RTX 5090
LLM：DeepSeek V4 Flash
运行时间：约 2 小时 8 分钟
```

### 4.2 主系统预测结果

```text
walk-forward 方向准确率：51.09%
正向事件准确率：47.37%
负向事件准确率：54.44%
平均 MAE：0.0158

对比：
v3 full-data（有泄漏）= 61.90%
v3 LOO = 47.62%
```

### 4.3 FinMAS-Risk 核心结果

> `【FinMAS-Risk / 风控中心实习】`

```text
T+5 组合实现 CAR：约 -0.04%
T+5 组合实现 MDD：约 2.16%
T+5 组合实现 ES：约 1.11%

VaR 违约率：
  T+1：9.35%
  T+5：12.15%
  T+20：19.00%
理论违约率：5%

结论：
事件级 VaR 在长周期上明显低估尾部风险，
Kupiec 检验在 1/5/20 期限均被拒绝。
```

压力测试组合盈亏：

```text
2020 疫情情景：约 -9.41%
2025 关税情景：约 -6.50%
市场单日 -5%：约 -3.81%
市场单日 -10%：约 -7.62%
```

风险归因中贡献最大的行业：

```text
801790：约 20.08%
801180：约 14.21%
801750：约 12.23%
801780：约 10.61%
801080：约 8.76%
```

---

## 五、技术栈

```text
语言：Python
框架：PyTorch、FastAPI、Uvicorn
LLM：DeepSeek V4 Flash、Ollama
RAG：BGE-large-zh-v1.5、FAISS、BM25
时序模型：Informer、KF-Transformer、GatedFusion
数据：pandas、NumPy
可视化：Plotly
实验：walk-forward、GPU 并发、断点续跑
```

---

## 六、简历填法建议

### 主项目简历条目

```text
FinMAS：基于 LLM 的多智能体金融分析系统

- 设计并实现多智能体金融事件分析框架，融合机理 Agent、RAG、Informer 和 KF-Transformer。
- 通过 walk-forward 时间防火墙实现无未来泄漏回测，覆盖 321 个事件-行业场景。
- 在 RTX 5090 上完成 GPU 全量实验，最终方向准确率 51.09%。
```

### 风控中心实习条目

> `【FinMAS-Risk / 风控中心实习】`

```text
FinMAS-Risk：金融事件风险推演与监控模块

- 在金融科技风险监控中心负责完善风险评估模块，开发 FinMAS-Risk 离线风控服务。
- 构建事件级、行业级、组合级风险视图，实现 T+1/T+5/T+20 多期限 VaR、ES、MDD 和尾概率监控。
- 搭建 VaR 回测与校准体系，发现长周期尾部风险低估问题，T+20 违约率达 19.00%。
- 实现历史情景与市场冲击压力测试，支持风险限额、集中度和风险来源定位。
- 通过 FastAPI 和 Plotly 输出可审计风控报告，并接入 DeepSeek 自动生成风险点评。
```

---

## 七、风控中心相关部分标识

本文中所有带有以下标记的内容，优先用于风控中心实习经历：

```text
【FinMAS-Risk / 风控中心实习】
```

对应核心模块：

```text
risk_center/
  risk_models.py
  portfolio_risk.py
  stress_test.py
  attribution.py
  llm_commentary.py
  report.py
  api.py
```
