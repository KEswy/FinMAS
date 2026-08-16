"""Hybrid retrieval without silent random fallback.

Uses BM25 and TF-IDF vector similarity. FAISS is optional and only used when
available; otherwise the same dense feature matrix is used with NumPy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..data.time_firewall import TimeFirewall
from ..schemas import EvidenceChunk, EvidencePackage


def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    stop = {"的", "了", "在", "是", "和", "与", "对", "为", "将", "等", "也", "都", "但", "及"}
    return [t for t in tokens if t not in stop and len(t) > 1]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: List[EvidenceChunk] = []
        self._tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._avgdl = 0.0

    def index(self, chunks: Sequence[EvidenceChunk]) -> None:
        self._chunks = list(chunks)
        self._tf = []
        self._df = {}
        total = 0
        for chunk in self._chunks:
            tokens = _tokenize(chunk.text)
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1
            self._tf.append(tf)
            total += len(tokens)
        self._avgdl = total / max(len(self._tf), 1)

    def search(self, query: str, top_k: int) -> List[EvidenceChunk]:
        n = len(self._chunks)
        if n == 0:
            return []
        q = _tokenize(query)
        scored = []
        for i, (chunk, tf) in enumerate(zip(self._chunks, self._tf)):
            dl = sum(tf.values())
            score = 0.0
            for token in q:
                if token not in tf:
                    continue
                idf = math.log((n - self._df.get(token, 0) + 0.5) /
                               (self._df.get(token, 0) + 0.5) + 1.0)
                tf_val = tf[token] * (self.k1 + 1.0) / (
                    tf[token] + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-8))
                )
                score += idf * tf_val
            scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        return [self._chunks[i] for _, i in scored[:top_k]]

    def score(self, query: str, chunk: EvidenceChunk) -> float:
        """Return BM25 score for one chunk, useful for candidate re-ranking."""
        n = len(self._chunks)
        if n == 0:
            return 0.0
        try:
            index = self._chunks.index(chunk)
        except ValueError:
            return 0.0
        q = _tokenize(query)
        tf = self._tf[index]
        dl = sum(tf.values())
        score = 0.0
        for token in q:
            if token not in tf:
                continue
            idf = math.log((n - self._df.get(token, 0) + 0.5) /
                           (self._df.get(token, 0) + 0.5) + 1.0)
            tf_val = tf[token] * (self.k1 + 1.0) / (
                tf[token] + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-8))
            )
            score += idf * tf_val
        return score


class TfidfVectorIndex:
    def __init__(self) -> None:
        self._chunks: List[EvidenceChunk] = []
        self._matrix: Optional[np.ndarray] = None
        self._vectorizer = None

    def index(self, chunks: Sequence[EvidenceChunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = None
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                tokenizer=_tokenize, token_pattern=None, lowercase=False, max_features=512
            )
            self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks]).toarray()
            norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
            self._matrix = self._matrix / np.maximum(norms, 1e-8)
        except Exception:
            self._vectorizer = None
            self._matrix = None

    def search(self, query: str, top_k: int) -> List[EvidenceChunk]:
        if not self._chunks or self._matrix is None or self._vectorizer is None:
            return []
        q = self._vectorizer.transform([query]).toarray()
        q = q / np.maximum(np.linalg.norm(q), 1e-8)
        scores = self._matrix @ q.ravel()
        top = np.argsort(-scores)[:top_k]
        return [self._chunks[int(i)] for i in top]


class SemanticVectorIndex:
    """Optional BGE + FAISS index with TF-IDF fallback."""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5") -> None:
        self.model_name = model_name
        self._chunks: List[EvidenceChunk] = []
        self._model = None
        self._index = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def index(self, chunks: Sequence[EvidenceChunk]) -> None:
        self._chunks = list(chunks)
        self._index = None
        if not self._chunks:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            self._model = SentenceTransformer(self.model_name)
            texts = [c.text for c in self._chunks]
            embs = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            ).astype("float32")
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embs)
            index.add(embs)
            self._index = index
            self._available = True
        except Exception:
            self._available = False
            self._index = None

    def search(self, query: str, top_k: int) -> List[EvidenceChunk]:
        if not self.available or self._index is None or not self._chunks:
            return []
        import faiss
        import numpy as np

        q = self._model.encode(
            ["为这个句子生成表示以用于检索相关文章：" + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(q)
        top_k = min(top_k, len(self._chunks))
        _, idx = self._index.search(q, top_k)
        return [self._chunks[int(i)] for i in idx[0]]

    def score(self, query: str, chunk: EvidenceChunk) -> float:
        if not self.available or self._model is None:
            return 0.0
        try:
            import faiss
            import numpy as np

            q = self._model.encode(
                ["为这个句子生成表示以用于检索相关文章：" + query],
                normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")
            d = self._model.encode(
                [chunk.text], normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")
            return float(np.dot(q[0], d[0]))
        except Exception:
            return 0.0


class CrossEncoderReranker:
    """Optional lightweight cross-encoder reranker."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large") -> None:
        self.model_name = model_name
        self._model = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._available = True
        except Exception:
            self._available = False

    def rerank(self, query: str, chunks: Sequence[EvidenceChunk],
               top_k: int) -> List[EvidenceChunk]:
        if not chunks:
            return []
        self.load()
        if not self.available or self._model is None:
            return list(chunks[:top_k])
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(chunks, scores), key=lambda x: -float(x[1])
        )
        return [c for c, _ in ranked[:top_k]]


@dataclass(slots=True)
class HybridRetriever:
    bm25: BM25Index = field(default_factory=BM25Index)
    dense: TfidfVectorIndex = field(default_factory=TfidfVectorIndex)
    semantic: SemanticVectorIndex = field(default_factory=SemanticVectorIndex)
    reranker: CrossEncoderReranker = field(default_factory=CrossEncoderReranker)

    def index(self, chunks: Sequence[EvidenceChunk]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self.semantic.index(chunks)

    @staticmethod
    def _rrf(
        lists: Sequence[List[EvidenceChunk]],
        k: int = 60,
        weights: Optional[Sequence[float]] = None,
    ) -> List[EvidenceChunk]:
        scores: Dict[str, float] = {}
        mapping: Dict[str, EvidenceChunk] = {}
        weights = list(weights or [1.0] * len(lists))
        for list_idx, ranked in enumerate(lists):
            for rank, chunk in enumerate(ranked, start=1):
                weight = float(weights[min(list_idx, len(weights) - 1)])
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
                mapping[chunk.chunk_id] = chunk
        return [mapping[cid] for cid in sorted(scores, key=lambda x: -scores[x])]


    def retrieve(self, query: str, top_k: int = 8,
                 as_of_date: Optional[str] = None) -> EvidencePackage:
        dense_results = self.semantic.search(query, top_k * 2)
        if not dense_results:
            dense_results = self.dense.search(query, top_k * 2)
        if self.semantic.available:
            candidates = self.semantic.search(query, top_k * 7)
            chunks = self.reranker.rerank(query, candidates, top_k)
            package = EvidencePackage(query=query, chunks=chunks)
            if as_of_date:
                package = TimeFirewall(as_of_date).filter_evidence(package)
            package.total_tokens_est = int(sum(len(c.text.split()) * 1.3 for c in package.chunks))
            return package
        else:
            candidates = self._rrf(
                [self.bm25.search(query, top_k * 2), dense_results]
            )
        chunks = self.reranker.rerank(query, candidates, top_k)
        package = EvidencePackage(query=query, chunks=chunks)
        if as_of_date:
            package = TimeFirewall(as_of_date).filter_evidence(package)
        package.total_tokens_est = int(sum(len(c.text.split()) * 1.3 for c in package.chunks))
        return package


def load_document_library(path: str = "data/events/doc_library.json") -> List[EvidenceChunk]:
    """Load document chunks from a legacy JSON document library."""
    import json
    import time

    p = Path(path)
    if not p.exists():
        return []
    records = json.loads(p.read_text(encoding="utf-8"))
    chunks = []
    for rec in records:
        pub_date = rec.get("pub_date") or rec.get("event_date")
        try:
            pub_ts = time.mktime(time.strptime(pub_date, "%Y-%m-%d"))
        except Exception:
            pub_ts = 0.0
        chunks.append(
            EvidenceChunk(
                chunk_id=str(rec.get("chunk_id", "")),
                text=str(rec.get("text", "")),
                source=str(rec.get("source", "未知")),
                pub_time=float(pub_ts),
                entities=[str(x) for x in (rec.get("entities") or [])],
                metadata={
                    "event_date": rec.get("event_date", ""),
                    "event_tag": rec.get("event_tag", ""),
                    "doc_type": rec.get("doc_type", ""),
                },
            )
        )
    return chunks


def build_retriever(path: str = "data/events/doc_library.json") -> HybridRetriever:
    retriever = HybridRetriever()
    retriever.index(load_document_library(path))
    return retriever
