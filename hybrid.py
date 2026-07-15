"""Hybrid fusion layer — combines sparse (TF-IDF) and dense (embedding) rankings.

Supports two fusion strategies:
  - weighted:  final_score = alpha * dense_score + (1 - alpha) * sparse_score
  - rrf:       Reciprocal Rank Fusion, score = sum(1 / (k + rank)) across retrievers
"""

from dense import DenseRetriever
from sparse import SparseRetriever, load_documents


def _normalize(scores):
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.sparse = SparseRetriever(documents)
        self.dense = DenseRetriever(documents)

    def _full_rankings(self, query):
        n = len(self.documents)
        sparse_ranked = self.sparse.search(query, top_k=n)
        dense_ranked = self.dense.search(query, top_k=n)
        return sparse_ranked, dense_ranked

    def search_weighted(self, query, top_k=5, alpha=0.5):
        sparse_ranked, dense_ranked = self._full_rankings(query)

        sparse_scores = {doc["id"]: score for doc, score in sparse_ranked}
        dense_scores = {doc["id"]: score for doc, score in dense_ranked}

        ids = list(sparse_scores.keys())
        sparse_norm = dict(zip(ids, _normalize([sparse_scores[i] for i in ids])))
        dense_norm = dict(zip(ids, _normalize([dense_scores[i] for i in ids])))

        fused = [
            (doc, alpha * dense_norm[doc["id"]] + (1 - alpha) * sparse_norm[doc["id"]])
            for doc in self.documents
        ]
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def search_rrf(self, query, top_k=5, k=60):
        sparse_ranked, dense_ranked = self._full_rankings(query)

        rrf_scores = {doc["id"]: 0.0 for doc in self.documents}
        for rank, (doc, _) in enumerate(sparse_ranked, start=1):
            rrf_scores[doc["id"]] += 1.0 / (k + rank)
        for rank, (doc, _) in enumerate(dense_ranked, start=1):
            rrf_scores[doc["id"]] += 1.0 / (k + rank)

        by_id = {doc["id"]: doc for doc in self.documents}
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(by_id[doc_id], score) for doc_id, score in fused[:top_k]]


if __name__ == "__main__":
    docs = load_documents()
    retriever = HybridRetriever(docs)

    for query in [
        "What does card FUT-23-091 belong to?",
        "Need a poacher who never misses in the box",
    ]:
        print(f"\nQuery: {query}")
        for doc, score in retriever.search_weighted(query, top_k=3):
            print(f"  [{score:.3f}] id={doc['id']}  {doc['text']}")
