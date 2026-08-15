"""
RAG 检索流水线
查询改写 → 多路召回（稀疏+密集+图） → 两级重排 → 层级压缩 → 证据包
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 嵌入模型单例（进程内只加载一次）
# ──────────────────────────────────────────────
class _EmbeddingModel:
    """
    懒加载的 BGE 嵌入模型单例。
    首次调用 get() 时下载并加载模型，后续调用复用同一实例。

    BGE 检索最佳实践：
      · 文档侧：encode_documents()  → 无前缀，直接编码原文
      · 查询侧：encode_queries()    → 加检索指令前缀，提升召回质量
    """
    _model = None
    _model_name: str = ""

    # BGE 中文检索查询前缀（官方推荐用于短查询 vs 长文档的非对称场景）
    # 注意：当文档本身较短时（<64 token），前缀可能导致相似度下降。
    # 待第三步换上真实 256-512 token 文档后，将此开关改为 True 并重测。
    _USE_QUERY_PREFIX: bool = False
    _QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    @classmethod
    def get(cls, model_name: str = ""):
        """返回已加载的模型实例，必要时触发加载。"""
        model_name = model_name or CONFIG.rag.embedding_model
        if cls._model is not None and cls._model_name == model_name:
            return cls  # 已加载，直接返回

        # 禁止 transformers 在后台尝试自动格式转换（避免 SSL 噪音）
        # 模型已下载到本地后，此设置不影响任何功能
        import os
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(f"[EmbeddingModel] 首次加载模型：{model_name} …")
        try:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
            dim = cls._model.get_embedding_dimension()
            logger.info(f"[EmbeddingModel] 加载完成，向量维度={dim}")
        except Exception as e_online:
            # 网络不可用时，尝试从本地缓存加载（模型已下载过即可）
            logger.warning(f"[EmbeddingModel] 在线加载失败（{e_online}），尝试本地缓存…")
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(model_name, local_files_only=True)
                cls._model_name = model_name
                dim = cls._model.get_embedding_dimension()
                logger.info(f"[EmbeddingModel] 本地缓存加载成功，向量维度={dim}")
            except Exception as e_local:
                logger.error(f"[EmbeddingModel] 本地缓存也失败：{e_local}")
                logger.warning("[EmbeddingModel] 回退到随机向量（仅限调试，请检查模型是否已下载）")
                cls._model = None
                cls._model_name = ""
        return cls

    @classmethod
    def encode_documents(cls, texts: List[str]) -> "np.ndarray":
        """编码文档（无前缀）。"""
        import numpy as np
        if cls._model is None:
            logger.warning("[EmbeddingModel] 模型未加载，返回随机向量")
            return np.random.randn(len(texts), 768).astype("float32")
        embs = cls._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embs.astype("float32")

    @classmethod
    def encode_queries(cls, texts: List[str]) -> "np.ndarray":
        """编码查询。_USE_QUERY_PREFIX=True 时加检索指令前缀（需长文档场景）。"""
        if cls._USE_QUERY_PREFIX:
            texts = [cls._QUERY_PREFIX + t for t in texts]
        return cls.encode_documents(texts)

    @classmethod
    def dim(cls) -> int:
        """返回当前模型的向量维度。"""
        if cls._model is None:
            return 768  # 回退默认值
        return cls._model.get_embedding_dimension()


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────
@dataclass
class EvidenceChunk:
    chunk_id: str
    text: str
    source: str
    pub_time: float                     # unix timestamp
    score: float = 0.0
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_days(self) -> float:
        return (time.time() - self.pub_time) / 86400

@dataclass
class EvidencePackage:
    """最终交付给 LLM 的证据包"""
    query: str
    chunks: List[EvidenceChunk]
    subgraph_summary: str = ""
    total_tokens_est: int = 0

    def to_context(self) -> str:
        lines = [f"[证据包] 查询：{self.query}\n"]
        if self.subgraph_summary:
            lines.append(self.subgraph_summary + "\n")
        lines.append("[文档证据]")
        for i, c in enumerate(self.chunks, 1):
            lines.append(
                f"[{i}] 来源={c.source}  时间={time.strftime('%Y-%m-%d', time.localtime(c.pub_time))}"
            )
            lines.append(c.text)
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 向量索引（FAISS 封装）
# ──────────────────────────────────────────────
class VectorIndex:
    def __init__(self):
        self._chunks: List[EvidenceChunk] = []
        self._faiss_index = None
        self._dim: int = 0      # 实际向量维度，首次 build 时确定

    def add(self, chunks: List[EvidenceChunk]) -> None:
        self._chunks.extend(chunks)
        self._faiss_index = None    # 有新文档时触发重建

    def _build_index(self) -> None:
        """编码所有文档并建立 FAISS 内积索引。"""
        try:
            import faiss
            import numpy as np

            # 预热模型（首次调用会触发下载）
            _EmbeddingModel.get()

            texts = [c.text for c in self._chunks]
            embs  = _EmbeddingModel.encode_documents(texts)  # (N, dim)

            self._dim = embs.shape[1]
            # IndexFlatIP + L2 归一化 = 余弦相似度检索
            index = faiss.IndexFlatIP(self._dim)
            faiss.normalize_L2(embs)
            index.add(embs)
            self._faiss_index = index
            logger.info(
                f"[VectorIndex] 索引重建完成：{len(self._chunks)} 条文档，"
                f"维度={self._dim}"
            )
        except ImportError:
            logger.warning("[VectorIndex] faiss 未安装，退化为随机召回")
            self._faiss_index = "mock"

    def search(self, query: str, top_k: int) -> List[EvidenceChunk]:
        if not self._chunks:
            return []
        if self._faiss_index is None:
            self._build_index()

        # FAISS 未安装时的降级路径
        if self._faiss_index == "mock":
            import random
            k = min(top_k, len(self._chunks))
            results = random.sample(self._chunks, k)
            for c in results:
                c.score = 0.5
            return results

        import faiss
        import numpy as np

        # 查询侧加检索前缀，提升召回质量
        q_emb = _EmbeddingModel.encode_queries([query])   # (1, dim)
        faiss.normalize_L2(q_emb)
        top_k = min(top_k, len(self._chunks))
        D, I  = self._faiss_index.search(q_emb, top_k)

        results = []
        for j, idx in enumerate(I[0]):
            if idx < len(self._chunks):
                c = self._chunks[idx]
                c.score = float(D[0][j])
                results.append(c)
        return results

    @staticmethod
    def _replace_score(c: EvidenceChunk, s: float) -> EvidenceChunk:
        c.score = s
        return c


# ──────────────────────────────────────────────
# 中文分词（jieba）工具函数
# ──────────────────────────────────────────────
def _tokenize(text: str) -> List[str]:
    """
    中文分词：优先使用 jieba，不可用时退回空格切分。
    去除单字停用词和纯数字 token，保留有语义的词元。
    """
    _STOPWORDS = {"的", "了", "在", "是", "和", "与", "对", "为", "将",
                  "等", "也", "都", "但", "及", "或", "由", "从", "其"}
    try:
        import jieba
        jieba.setLogLevel(60)   # 静默 jieba 初始化日志
        tokens = [t for t in jieba.cut(text) if len(t) > 1 and t not in _STOPWORDS]
    except ImportError:
        tokens = text.lower().split()
    return tokens


# ──────────────────────────────────────────────
# BM25 稀疏检索（jieba 中文分词版）
# ──────────────────────────────────────────────
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: List[EvidenceChunk] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._avgdl: float = 0.0

    def add(self, chunks: List[EvidenceChunk]) -> None:
        for c in chunks:
            tokens = _tokenize(c.text)          # ← jieba 中文分词
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t in set(tokens):
                self._df[t] = self._df.get(t, 0) + 1
            self._tf.append(tf)
            self._chunks.append(c)
        total = sum(len(tf) for tf in self._tf)
        self._avgdl = total / max(len(self._tf), 1)

    def search(self, query: str, top_k: int) -> List[EvidenceChunk]:
        import math
        N = len(self._chunks)
        if N == 0:
            return []
        q_tokens = _tokenize(query)             # ← jieba 中文分词
        scores = []
        for i, (c, tf) in enumerate(zip(self._chunks, self._tf)):
            dl = sum(tf.values())
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                idf = math.log(
                    (N - self._df.get(t, 0) + 0.5) /
                    (self._df.get(t, 0) + 0.5) + 1
                )
                tf_val = tf[t] * (self.k1 + 1) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                )
                score += idf * tf_val
            scores.append((score, i))
        scores.sort(reverse=True)
        result = []
        for s, i in scores[:top_k]:
            chunk = self._chunks[i]
            chunk.score = s
            result.append(chunk)
        return result


# ──────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────
def reciprocal_rank_fusion(
    lists: List[List[EvidenceChunk]], k: int = 60
) -> List[EvidenceChunk]:
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, EvidenceChunk] = {}
    for ranked_list in lists:
        for rank, chunk in enumerate(ranked_list, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk
    sorted_ids = sorted(scores, key=lambda x: -scores[x])
    result = []
    for cid in sorted_ids:
        c = chunk_map[cid]
        c.score = scores[cid]
        result.append(c)
    return result


# ──────────────────────────────────────────────
# Cross-Encoder 重排（Stub）
# ──────────────────────────────────────────────
def cross_encoder_rerank(
    query: str,
    chunks: List[EvidenceChunk],
    top_k: int,
    time_decay_days: float = 30.0,
) -> List[EvidenceChunk]:
    """
    真实实现：调用 cross-encoder/ms-marco-MiniLM-L-6-v2 等模型。
    此处 Stub：以 BM25 分数 × 时效衰减作为代理排序。
    """
    import math
    for c in chunks:
        age_factor = math.exp(-c.age_days() / time_decay_days)
        c.score = c.score * age_factor
    chunks.sort(key=lambda x: -x.score)
    return chunks[:top_k]


# ──────────────────────────────────────────────
# 主检索器
# ──────────────────────────────────────────────
class Retriever:
    def __init__(self, kg=None):
        self.vector_index = VectorIndex()
        self.bm25_index = BM25Index()
        self.kg = kg

    # ── 索引建立 ────────────────────────────────
    def index_documents(self, chunks: List[EvidenceChunk]) -> None:
        self.vector_index.add(chunks)
        self.bm25_index.add(chunks)

    # ── 查询改写（调用 LLM）────────────────────
    @staticmethod
    def rewrite_query(query: str, context: str = "") -> str:
        """
        改写查询以提升召回质量。
        生产实现调用 Claude；此处返回原始查询。
        """
        return query   # stub；实际替换为 LLM call

    # ── 主入口 ──────────────────────────────────
    def retrieve(
        self,
        query: str,
        entity_names: Optional[List[str]] = None,
        step_context: str = "",
    ) -> EvidencePackage:
        cfg = CONFIG.rag
        rewritten = self.rewrite_query(query, step_context)

        # 多路召回
        dense_results  = self.vector_index.search(rewritten, cfg.top_k_dense)
        sparse_results = self.bm25_index.search(rewritten, cfg.top_k_sparse)
        fused = reciprocal_rank_fusion([dense_results, sparse_results])

        # 两级重排（cross-encoder + 时效）
        reranked = cross_encoder_rerank(rewritten, fused, cfg.rerank_top_k)

        # 图检索
        subgraph_summary = ""
        if self.kg and entity_names:
            sg = self.kg.get_subgraph(entity_names)
            subgraph_summary = sg.to_summary()

        # 估算 token 数
        total_tokens = sum(len(c.text.split()) * 1.3 for c in reranked)

        return EvidencePackage(
            query=query,
            chunks=reranked,
            subgraph_summary=subgraph_summary,
            total_tokens_est=int(total_tokens),
        )

    # ── 从文档库 JSON 加载（替代 build_demo）──────
    @classmethod
    def build_from_library(cls, path: str = "data/events/doc_library.json",
                           kg=None) -> "Retriever":
        """
        从 data/events/doc_library.json 加载真实金融文档语料，
        构建向量索引（BGE 嵌入）和 BM25 索引（jieba 分词）。

        fallback: 若文件不存在则退回 build_demo()，保证系统可运行。
        DOC_LIBRARY_PATH env var lets RAG ablations swap the corpus
        (e.g. 74-doc no-mechanism vs 92-doc with-mechanism) without code edits.
        """
        import json as _json
        import os as _os
        path = _os.environ.get("DOC_LIBRARY_PATH", path)
        try:
            with open(path, encoding="utf-8") as f:
                records = _json.load(f)
            logger.info(f"[Retriever] 加载文档库: {path} ({len(records)} 篇)")
        except FileNotFoundError:
            logger.warning(f"[Retriever] 文档库未找到（{path}），退回演示模式")
            return cls.build_demo(kg=kg)

        r = cls(kg=kg)
        chunks = []
        import time as _time
        for rec in records:
            try:
                pub_ts = _time.mktime(
                    _time.strptime(rec.get("pub_date", rec["event_date"]), "%Y-%m-%d")
                )
            except Exception:
                pub_ts = _time.time()

            chunks.append(EvidenceChunk(
                chunk_id = rec["chunk_id"],
                text     = rec["text"],
                source   = rec.get("source", "未知"),
                pub_time = pub_ts,
                entities = rec.get("entities", []),
                metadata = {
                    "event_date": rec.get("event_date", ""),
                    "event_tag":  rec.get("event_tag", ""),
                    "doc_type":   rec.get("doc_type", ""),
                },
            ))

        r.index_documents(chunks)
        logger.info(
            f"[Retriever] 已从 {path} 加载 {len(chunks)} 条文档，"
            f"覆盖 {len({c.metadata['event_tag'] for c in chunks})} 个事件"
        )
        return r

    # ── 构建示例文档库（保留用于快速调试）────────
    @classmethod
    def build_demo(cls, kg=None) -> "Retriever":
        r = cls(kg=kg)
        demo_chunks = [
            EvidenceChunk(
                "c001", "央行宣布降息50个基点，为近年最大幅度降息，旨在刺激经济增长。",
                "新华社", pub_time=time.time()-3600, entities=["央行","降息"]
            ),
            EvidenceChunk(
                "c002", "银行业将直接受益于宽松货币政策，净息差有望扩大，贷款需求或提升。",
                "中信证券研报", pub_time=time.time()-7200, entities=["银行业","净息差"]
            ),
            EvidenceChunk(
                "c003", "房地产行业融资成本下降，有助于缓解流动性压力，但去库存压力仍存。",
                "申万宏源研报", pub_time=time.time()-14400, entities=["房地产","融资成本"]
            ),
            EvidenceChunk(
                "c004", "招商银行作为全国性股份制银行，对利率政策敏感性较高，历史上降息后股价平均上涨3-5%。",
                "Wind金融终端", pub_time=time.time()-86400, entities=["招商银行","利率敏感"]
            ),
        ]
        r.index_documents(demo_chunks)
        return r
