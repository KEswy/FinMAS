"""
机理型 Agent (Mechanism Agent)
职责：证据约束的 CoT 推理 + 跨层传导 + 机理链 JSON 生成
"""
from __future__ import annotations
import json, time, logging, os

from typing import Dict, List, Optional, Any

from config import CONFIG
from rag.retrieval import Retriever, EvidencePackage
from rag.knowledge_graph import KnowledgeGraph
from rag.prompts import (
    SYSTEM_PROMPT, STEP_REASONING_TEMPLATE,
    PROPAGATION_TEMPLATE, REVERSE_RETRIEVAL_TEMPLATE,
    TASK_TEMPLATES
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 机理链数据结构
# ──────────────────────────────────────────────
class MechanismChain:
    def __init__(self, raw: Dict):
        self.raw = raw
        self.steps: List[Dict] = raw.get("chain", [])
        self.overall_confidence: float = raw.get("overall_confidence", 0.0)
        self.evidence_gaps: List[str] = raw.get("evidence_gaps", [])
        self.recommendation: str = raw.get("recommendation", "")

    def to_exogenous_factors(self) -> Dict[str, float]:
        """从机理链中提取外生因子（供数据 Agent 使用）"""
        factors: Dict[str, float] = {}
        for step in self.steps:
            entity = step.get("entity", "")
            direction = 1.0 if step.get("impact_direction") == "+" else -1.0
            mag_map = {"高": 0.9, "中": 0.5, "低": 0.2, "未知": 0.0}
            magnitude = mag_map.get(step.get("impact_magnitude", "未知"), 0.0)
            factors[entity] = direction * magnitude * step.get("confidence", 0.5)
        return factors

    def consistency_score(self) -> float:
        """简单一致性分数：置信度均值"""
        if not self.steps:
            return 0.0
        return sum(s.get("confidence", 0) for s in self.steps) / len(self.steps)

    def to_json(self) -> str:
        return json.dumps(self.raw, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────
# 机理 Agent
# ──────────────────────────────────────────────
class MechanismAgent:
    # def __init__(
    #     self,
    #     retriever: Retriever,
    #     kg: KnowledgeGraph,
    #     api_key: Optional[str] = None,
    # ):
    #     self.retriever = retriever
    #     self.kg = kg
    #     self.client = anthropic.Anthropic(
    #         api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    #     )
    #     self.cfg = CONFIG.llm

    def __init__(
            self,
            retriever: Retriever,
            kg: KnowledgeGraph,
            api_key: Optional[str] = None,
            blind_mode: bool = False,
    ):
        self.retriever = retriever
        self.kg = kg
        self.ollama_url = "http://localhost:11434/api/chat"
        # LLM_MODEL env var lets cross-LLM ablations switch model without code edits
        # (e.g. LLM_MODEL=llama3.1:8b, LLM_MODEL=qwen2.5:14b). Default: qwen2.5:32b.
        import os as _os
        self.model_name = _os.environ.get("LLM_MODEL", "qwen2.5:32b")
        self.cfg = CONFIG.llm
        # blind_mode=True 时完全屏蔽方向先验，用于无标签泄漏的评测
        self.blind_mode = blind_mode
        # 生产模式先验权重=1（弱参考）；盲测模式=0（纯推理）
        self.hint_weight = 0 if blind_mode else 1

    # ── LLM 调用（带重试 + 服务可用性检查） ────────────────
    def _call_llm(self, user_msg: str, system: str = SYSTEM_PROMPT) -> str:
        """
        指数退避重试：5s -> 10s -> 20s -> 40s -> 60s -> 60s -> 60s -> 60s -> 60s -> 60s
        总等待预算 ~6 分钟，足够 Ollama 服务从 OOM-kill 后自动重启

        DeepSeek-R1 类模型会在响应开头输出 <think>...</think> 推理块，
        这里统一剥离，让下游 JSON / regex 解析不会被推理块干扰。
        """
        import urllib.request, urllib.error, json as _json, time as _time, re as _re
        payload = _json.dumps({
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            # keep_alive=60m: 让模型保持在显存里 1 小时不被卸载，
            # 避免长时间运行中频繁冷加载触发的 Ollama Windows 偶发挂掉
            "keep_alive": "60m",
            "options": {"temperature": self.cfg.temperature},
        }).encode("utf-8")

        backoff = [5, 10, 20, 40, 60, 60, 60, 60, 60, 60]
        last_err: Exception | None = None
        for attempt, wait in enumerate(backoff, start=1):
            try:
                req = urllib.request.Request(
                    self.ollama_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req, timeout=180)
                result = _json.loads(resp.read())
                content = result["message"]["content"]
                # Strip <think>...</think> reasoning blocks (DeepSeek-R1 et al.)
                content = _re.sub(r"<think>.*?</think>", "", content,
                                  flags=_re.DOTALL).strip()
                return content
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last_err = e
                print(f"    [LLM retry {attempt}/{len(backoff)}] {type(e).__name__}: {e}; "
                      f"waiting {wait}s before retry")
                _time.sleep(wait)
                # Sanity-ping the server before the next chat call
                try:
                    ping = urllib.request.Request(
                        self.ollama_url.replace("/api/chat", "/api/tags"))
                    urllib.request.urlopen(ping, timeout=10).read()
                    print(f"    [LLM retry {attempt}] /api/tags reachable, retrying chat")
                except Exception as pe:
                    print(f"    [LLM retry {attempt}] /api/tags still unreachable: {pe}")
        raise RuntimeError(f"Ollama unreachable after {len(backoff)} retries: {last_err}")
    # def _call_llm(self, user_msg: str, system: str = SYSTEM_PROMPT) -> str:
    #     # ── Mock 模式：无需 API Key ──────────────────
    #     import json
    #     mock_chain = {
    #         "event_summary": "央行降息50bp，宽松货币政策预期升温。",
    #         "chain": [
    #             {
    #                 "step": 1, "layer": "social", "entity": "央行降息政策",
    #                 "reasoning": "央行降息50bp直接释放宽松信号，市场流动性预期改善。",
    #                 "evidence_ids": ["c001"], "confidence": 0.90,
    #                 "impact_direction": "+", "impact_magnitude": "高"
    #             },
    #             {
    #                 "step": 2, "layer": "industry", "entity": "银行业",
    #                 "reasoning": "降息直接扩大银行息差空间，贷款需求有望提升。",
    #                 "evidence_ids": ["c002"], "confidence": 0.82,
    #                 "impact_direction": "+", "impact_magnitude": "高"
    #             },
    #             {
    #                 "step": 3, "layer": "company", "entity": "招商银行",
    #                 "reasoning": "招商银行零售业务占比高，对利率政策敏感性强。",
    #                 "evidence_ids": ["c004"], "confidence": 0.78,
    #                 "impact_direction": "+", "impact_magnitude": "中"
    #             },
    #             {
    #                 "step": 4, "layer": "asset", "entity": "招商银行股价",
    #                 "reasoning": "基本面改善预期驱动股价短期上行，历史降息后平均涨幅3-5%。",
    #                 "evidence_ids": ["c004"], "confidence": 0.72,
    #                 "impact_direction": "+", "impact_magnitude": "中"
    #             }
    #         ],
    #         "overall_confidence": 0.80,
    #         "evidence_gaps": ["缺乏最新利率敏感性数据"],
    #         "recommendation": "降息利好银行及房地产行业，短期关注招商银行和万科A的超额收益机会。"
    #     }
    #     return json.dumps(mock_chain, ensure_ascii=False)
    # ── 单步 RAT 推理 ────────────────────────────
    def _rat_step(
        self,
        step_num: int,
        total_steps: int,
        current_layer: str,
        sub_question: str,
        prev_chain: str,
        entity_names: List[str],
        direction_hint: str = "",
        direction_hint_text: str = "",
        hint_weight: int = 1,
    ) -> Dict:
        # 按需检索
        pkg = self.retriever.retrieve(sub_question, entity_names=entity_names, step_context=prev_chain)

        prompt = STEP_REASONING_TEMPLATE.format(
            step_num=step_num,
            total_steps=total_steps,
            current_layer=current_layer,
            sub_question=sub_question,
            prev_chain=prev_chain or "（无）",
            evidence_context=pkg.to_context(),
            kg_context=pkg.subgraph_summary or "（无图谱信息）",
        )
        # 注入方向提示
        if direction_hint_text:
            prompt = direction_hint_text + prompt

        raw_response = self._call_llm(prompt)

        # 解析（简化：把响应包装为步骤 dict）
        return {
            "step": step_num,
            "layer": current_layer,
            "sub_question": sub_question,
            "response": raw_response,
            "evidence_ids": [c.chunk_id for c in pkg.chunks],
            "confidence": self._extract_confidence(raw_response),
            "impact_direction": self._extract_direction(raw_response, direction_hint, hint_weight),
            "impact_magnitude": self._extract_magnitude(raw_response),
        }

    # ── 主推理入口 ───────────────────────────────
    def analyze(
        self,
        event_description: str,
        task_type: str = "policy_event",
        entity_names: Optional[List[str]] = None,
        event_direction: str = "",
    ) -> MechanismChain:
        """
        完整的 RAT×KG 推理。

        方向先验来源（优先级递减）：
          1. 生产模式下的外部标注 event_direction（已验证的外部信号）
          2. 从事件文本关键词自动推断（盲测/生产均适用，不依赖标签）
          3. LLM 自主推理结论
        """
        entity_names = entity_names or []
        sub_questions = TASK_TEMPLATES.get(task_type, TASK_TEMPLATES["policy_event"])["sub_questions"]
        layer_seq = ["social", "industry", "company", "asset", "asset"]
        total_steps = len(sub_questions)

        # ── 方向推断 ─────────────────────────────
        # Step 1：生产模式下使用外部标注方向
        effective_dir = ""
        if not self.blind_mode and event_direction:
            effective_dir = event_direction

        # Step 2：从事件文本关键词推断（盲测/生产均适用）
        # 这不是数据泄漏——事件描述是已知输入，不依赖 CAR 标签
        if not effective_dir:
            effective_dir = self._infer_direction_from_text(event_description)
            if effective_dir:
                logger.info(f"[MechanismAgent] 文本推断方向={effective_dir}（关键词匹配）")

        # 有推断方向时注入提示（hint_weight 根据来源差异化）
        # 外部标注：weight=2（较强参考）；文本推断：weight=1（弱参考）
        if not self.blind_mode and event_direction:
            active_weight = 2
        elif effective_dir:
            active_weight = 1
        else:
            active_weight = 0

        steps_log = []
        prev_chain_text = ""

        # 方向提示文字
        direction_hint_text = ""
        if effective_dir == "-":
            direction_hint_text = "\n【重要提示】该事件整体为利空/负面冲击，请重点分析负面影响路径，评估下跌风险。\n"
        elif effective_dir == "+":
            direction_hint_text = "\n【重要提示】该事件整体为利好/正面刺激，请重点分析正面影响路径，评估上涨机会。\n"

        mode_tag = f"盲测(文本推断={effective_dir or '无'})" if self.blind_mode \
                   else f"生产(先验={event_direction or '无'}/文本={effective_dir or '无'})"
        logger.info(f"[MechanismAgent] 开始分析：{event_description[:50]}… 模式={mode_tag}")

        # 图谱传导计算
        impact_scores = {}
        if entity_names and self.kg:
            sg = self.kg.get_subgraph(entity_names)
            if sg.nodes:
                seed_id = sg.nodes[0].node_id
                impact_scores = self.kg.propagate_impact(seed_id)

        # RAT 步进推理
        for i, sub_q in enumerate(sub_questions, 1):
            layer = layer_seq[min(i-1, len(layer_seq)-1)]
            step_result = self._rat_step(
                step_num=i,
                total_steps=total_steps,
                current_layer=layer,
                sub_question=sub_q,
                prev_chain=prev_chain_text,
                entity_names=entity_names,
                direction_hint=effective_dir,
                direction_hint_text=direction_hint_text,
                hint_weight=active_weight,
            )
            steps_log.append(step_result)
            prev_chain_text += f"\n步骤{i}（{layer}层）：{step_result['response'][:200]}…"

        # 汇总机理链（调用 LLM 综合）
        summary_prompt = f"""
事件：{event_description}
{direction_hint_text}
图谱传导分数：{json.dumps(impact_scores, ensure_ascii=False)}
各步推理摘要：
{prev_chain_text}

请按照规定的 JSON 格式，输出完整的机理链（chain）和综合结论。
注意：impact_direction 必须准确反映该事件对各层的实际影响方向（"+"为正面，"-"为负面）。
只输出 JSON，不要其他内容。
"""
        raw_json = self._call_llm(summary_prompt)

        # 解析 JSON
        try:
            # 提取 JSON 块
            start = raw_json.find("{")
            end = raw_json.rfind("}") + 1
            chain_dict = json.loads(raw_json[start:end])
        except Exception:
            # 降级：包装步骤日志
            chain_dict = {
                "event_summary": event_description,
                "chain": [
                    {
                        "step": s["step"],
                        "layer": s["layer"],
                        "entity": entity_names[0] if entity_names else "未知",
                        "reasoning": s["response"],
                        "evidence_ids": s["evidence_ids"],
                        "confidence": s["confidence"],
                        "impact_direction": s["impact_direction"],
                        "impact_magnitude": s["impact_magnitude"],
                    }
                    for s in steps_log
                ],
                "overall_confidence": sum(s["confidence"] for s in steps_log) / max(len(steps_log), 1),
                "evidence_gaps": [],
                "recommendation": "（综合分析见各步骤）",
            }

        return MechanismChain(chain_dict)

    # ── 残差触发逆向取证 ─────────────────────────
    def reverse_retrieval(
        self,
        residual_info: str,
        time_window: str,
        affected_assets: List[str],
        chain: MechanismChain,
    ) -> MechanismChain:
        """残差超阈值时调用，重新取证并更新机理链"""
        prompt = REVERSE_RETRIEVAL_TEMPLATE.format(
            residual_info=residual_info,
            time_window=time_window,
            affected_assets=", ".join(affected_assets),
            chain_gaps=", ".join(chain.evidence_gaps) or "未知",
            evidence_gaps="请补充该时段内的政策变动、突发事件等信息。",
        )
        updated_response = self._call_llm(prompt)
        logger.info(f"[MechanismAgent] 逆向取证完成，更新机理链")

        # 简化：在原链上追加修正步骤
        chain.raw.setdefault("corrections", []).append({
            "triggered_by": "residual",
            "residual_info": residual_info,
            "time_window": time_window,
            "response": updated_response,
            "timestamp": time.time(),
        })
        return chain

    # ── 工具方法 ────────────────────────────────
    @staticmethod
    def _extract_confidence(text: str) -> float:
        import re
        m = re.search(r"置信度[：:]\s*(0\.\d+|1\.0)", text)
        if m:
            return float(m.group(1))
        m = re.search(r"(0\.\d+)", text)
        return float(m.group(1)) if m else 0.5

    @staticmethod
    def _infer_direction_from_text(text: str) -> str:
        """
        从事件文本关键词推断方向（不依赖标签，盲测合法）。

        适用逻辑：
          · 负向词命中数 > 正向词 → 返回 "-"
          · 正向词命中数 > 负向词 → 返回 "+"
          · 平局或均为0 → 返回 ""（交给 LLM 自主判断）
        """
        NEG_TERMS = [
            "关税", "制裁", "加征", "报复", "反制",    # 贸易摩擦
            "暴跌", "大跌", "崩盘", "熔断", "黑色",    # 市场崩跌
            "双减", "整顿", "打压", "处罚", "罚款",    # 监管收紧
            "违约", "退市", "爆雷", "破产", "债务",    # 信用事件
            "收紧", "禁止", "限制上市", "冻结",        # 政策限制
        ]
        POS_TERMS = [
            "降息", "降准", "宽松", "减息",            # 货币宽松
            "降至", "下调", "一揽子", "组合拳",        # 降息/政策包
            "减半", "减免", "补贴", "扶持",            # 政策红利
            "活跃", "刺激", "提振", "托底",            # 市场提振
            "利好", "松绑", "放开", "支持",            # 正面政策
            "里程碑", "突破", "登顶", "超预期",        # 正面事件
        ]
        neg = sum(1 for t in NEG_TERMS if t in text)
        pos = sum(1 for t in POS_TERMS if t in text)

        if neg > pos:
            return "-"
        if pos > neg:
            return "+"
        return ""   # 平局：让 LLM 自主判断，不强制默认正向

    @staticmethod
    def _extract_direction(text: str, direction_hint: str = "",
                           hint_weight: int = 1) -> str:
        """
        基于关键词计分的方向检测。
        direction_hint: 来自事件库或外部信号的方向先验（"+"或"-"）。
        hint_weight:    先验票权重。
          - 生产模式（默认）: hint_weight=1，先验作为弱参考，不主导结论。
          - 盲测模式:         hint_weight=0（不传 direction_hint），纯靠 LLM 推理。
          - 旧行为（已废弃）: hint_weight=3，过强，导致泄漏，不应再使用。
        """
        pos_words = ["利好", "正面", "上涨", "上行", "受益", "提振", "宽松", "积极", "改善", "增长"]
        neg_words = ["利空", "负面", "下跌", "下行", "承压", "冲击", "收紧", "风险", "恶化", "下降",
                     "打压", "重挫", "暴跌", "拖累", "萎缩", "衰退", "关税", "制裁", "限制"]

        pos_count = sum(1 for w in pos_words if w in text)
        neg_count = sum(1 for w in neg_words if w in text)

        # 先验权重：生产模式=1（弱参考），盲测模式=0（不使用）
        if hint_weight > 0:
            if direction_hint == "+":
                pos_count += hint_weight
            elif direction_hint == "-":
                neg_count += hint_weight

        if neg_count > pos_count:
            return "-"
        elif pos_count > neg_count:
            return "+"
        # 平局时：有先验则采用先验，否则返回 "?" 由上层的综合摘要决定
        if hint_weight > 0 and direction_hint in ("+", "-"):
            return direction_hint
        return "?"  # 真平局且无先验：不强制正向

    @staticmethod
    def _extract_magnitude(text: str) -> str:
        for word in ["高", "显著", "大幅"]:
            if word in text:
                return "高"
        for word in ["中", "温和", "一定程度"]:
            if word in text:
                return "中"
        for word in ["低", "有限", "轻微"]:
            if word in text:
                return "低"
        return "未知"
