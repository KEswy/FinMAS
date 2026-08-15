"""
知识图谱管理 — 社会/行业/公司/资产四层结构
使用 Neo4j 作为图数据库后端；提供 Mock 模式用于无图库时测试。
"""
from __future__ import annotations
import json, math, time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from config import CONFIG

# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────
@dataclass
class KGNode:
    node_id: str
    layer: str                     # social | industry | company | asset
    name: str
    node_type: str                 # event | policy | sector | stock | index …
    attributes: Dict[str, Any] = field(default_factory=dict)
    # 关联证据片段 id
    evidence_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

@dataclass
class KGEdge:
    src: str
    dst: str
    relation: str                  # influence | cause | supply_chain | belongs_to …
    weight: float = 1.0
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    # 时效衰减：有效期（秒），None 表示永久
    ttl: Optional[float] = None

    def effective_weight(self) -> float:
        """衰减后的有效边权"""
        if self.ttl is None:
            return self.weight
        age_days = (time.time() - self.created_at) / 86400
        half_life = CONFIG.kg.decay_half_life
        decay = math.exp(-math.log(2) * age_days / half_life)
        return self.weight * decay * self.confidence

@dataclass
class SubGraph:
    nodes: List[KGNode]
    edges: List[KGEdge]

    def to_summary(self) -> str:
        """生成供 LLM 阅读的文本摘要"""
        lines = ["[知识图谱子图摘要]"]
        for n in self.nodes:
            lines.append(f"  节点 [{n.layer}] {n.name}（{n.node_type}）")
        for e in self.edges:
            lines.append(
                f"  {e.src} --[{e.relation}, w={e.effective_weight():.3f}]--> {e.dst}"
            )
        return "\n".join(lines)

# ──────────────────────────────────────────────
# 图数据库接口
# ──────────────────────────────────────────────
class KnowledgeGraph:
    """
    统一知识图谱接口。
    use_mock=True 时使用内存字典，方便无 Neo4j 环境调试。
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        # Mock 存储
        self._nodes: Dict[str, KGNode] = {}
        self._edges: List[KGEdge] = []
        if not use_mock:
            self._init_neo4j()

    # ── 初始化 ──────────────────────────────────
    def _init_neo4j(self):
        try:
            from neo4j import GraphDatabase
            cfg = CONFIG.rag
            self._driver = GraphDatabase.driver(
                cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
            )
        except ImportError:
            raise RuntimeError("请先 pip install neo4j")

    # ── 写入 ────────────────────────────────────
    def add_node(self, node: KGNode) -> None:
        if self.use_mock:
            self._nodes[node.node_id] = node
        else:
            with self._driver.session() as s:
                s.run(
                    "MERGE (n:Entity {node_id: $id}) SET n += $props",
                    id=node.node_id,
                    props={**asdict(node), "layer": node.layer},
                )

    def add_edge(self, edge: KGEdge) -> None:
        if self.use_mock:
            self._edges.append(edge)
        else:
            with self._driver.session() as s:
                s.run(
                    """
                    MATCH (a:Entity {node_id:$src}), (b:Entity {node_id:$dst})
                    MERGE (a)-[r:RELATION {relation:$rel}]->(b)
                    SET r.weight=$w, r.confidence=$c, r.created_at=$ca
                    """,
                    src=edge.src, dst=edge.dst, rel=edge.relation,
                    w=edge.weight, c=edge.confidence, ca=edge.created_at,
                )

    # ── 查询 ────────────────────────────────────
    def get_subgraph(
        self,
        entity_names: List[str],
        max_hops: int = None,
    ) -> SubGraph:
        """以实体名为种子，返回 k-hop 子图"""
        max_hops = max_hops or CONFIG.kg.max_hops
        thr = CONFIG.kg.edge_weight_threshold

        if self.use_mock:
            # 找到种子节点
            seed_ids = {
                nid for nid, n in self._nodes.items()
                if any(name in n.name for name in entity_names)
            }
            visited = set(seed_ids)
            frontier = set(seed_ids)
            for _ in range(max_hops):
                next_frontier = set()
                for e in self._edges:
                    if e.effective_weight() < thr:
                        continue
                    if e.src in frontier and e.dst not in visited:
                        next_frontier.add(e.dst)
                    if e.dst in frontier and e.src not in visited:
                        next_frontier.add(e.src)
                visited |= next_frontier
                frontier = next_frontier
            nodes = [self._nodes[nid] for nid in visited if nid in self._nodes]
            edges = [
                e for e in self._edges
                if e.src in visited and e.dst in visited
                and e.effective_weight() >= thr
            ]
            return SubGraph(nodes=nodes, edges=edges)
        else:
            raise NotImplementedError("Neo4j 子图查询待实现")

    # ── 传导计算 ─────────────────────────────────
    def propagate_impact(
        self,
        event_node_id: str,
        initial_strength: float = 1.0,
    ) -> Dict[str, float]:
        """
        从事件节点沿边传导，计算各节点的影响分数。
        返回 {node_id: impact_score}
        """
        scores: Dict[str, float] = {event_node_id: initial_strength}
        queue = [(event_node_id, initial_strength)]
        visited = {event_node_id}
        layer_order = CONFIG.kg.layers  # social→industry→company→asset

        while queue:
            src_id, strength = queue.pop(0)
            src_node = self._nodes.get(src_id)
            if src_node is None:
                continue
            src_layer_idx = (
                layer_order.index(src_node.layer)
                if src_node.layer in layer_order else -1
            )
            for edge in self._edges:
                if edge.src != src_id:
                    continue
                dst_node = self._nodes.get(edge.dst)
                if dst_node is None or edge.dst in visited:
                    continue
                dst_layer_idx = (
                    layer_order.index(dst_node.layer)
                    if dst_node.layer in layer_order else -1
                )
                # 只沿层级方向传导（或同层）
                if dst_layer_idx < src_layer_idx:
                    continue
                new_strength = strength * edge.effective_weight()
                if new_strength < CONFIG.kg.edge_weight_threshold:
                    continue
                scores[edge.dst] = scores.get(edge.dst, 0) + new_strength
                visited.add(edge.dst)
                queue.append((edge.dst, new_strength))

        return scores

    # ── 边权更新（残差触发） ──────────────────────
    def update_edge_weight(
        self, src: str, dst: str, relation: str, delta: float
    ) -> None:
        """根据残差信号小幅调整边权"""
        for e in self._edges:
            if e.src == src and e.dst == dst and e.relation == relation:
                e.weight = max(0.0, min(2.0, e.weight + delta))
                e.created_at = time.time()  # 刷新时效

    # ── 工厂方法：构建示例图谱 ───────────────────
    @classmethod
    def build_demo(cls) -> "KnowledgeGraph":
        kg = cls(use_mock=True)
        # 节点
        nodes = [
            KGNode("ev_001", "social",   "央行降息50bp",    "policy_event"),
            KGNode("ind_001","industry", "银行业",          "sector"),
            KGNode("ind_002","industry", "房地产业",        "sector"),
            KGNode("co_001", "company",  "招商银行",        "stock"),
            KGNode("co_002", "company",  "万科A",           "stock"),
            KGNode("as_001", "asset",    "招商银行股价",    "price"),
            KGNode("as_002", "asset",    "万科A股价",       "price"),
        ]
        for n in nodes:
            kg.add_node(n)
        # 边
        edges = [
            KGEdge("ev_001", "ind_001", "influence", weight=0.8, confidence=0.9),
            KGEdge("ev_001", "ind_002", "influence", weight=0.7, confidence=0.85),
            KGEdge("ind_001","co_001",  "belongs_to",weight=0.9, confidence=1.0),
            KGEdge("ind_002","co_002",  "belongs_to",weight=0.9, confidence=1.0),
            KGEdge("co_001", "as_001",  "determines",weight=1.0, confidence=1.0),
            KGEdge("co_002", "as_002",  "determines",weight=1.0, confidence=1.0),
        ]
        for e in edges:
            kg.add_edge(e)
        return kg
