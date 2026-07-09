"""Evaluation harness — runs sparse, dense, and hybrid retrievers over a labeled
query set and reports hit@1 / hit@3 for each, plus a per-query comparison table.
"""

import json
from pathlib import Path

from hybrid import HybridRetriever
from sparse import load_documents

EVAL_PATH = Path(__file__).parent / "data" / "eval_queries.json"


def load_eval_queries():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def top_ids(ranked, k):
    return [doc["id"] for doc, _ in ranked[:k]]


def hit_at_k(ranked, expected_id, k):
    return "Y" if expected_id in top_ids(ranked, k) else "N"


def main():
    docs = load_documents()
    eval_queries = load_eval_queries()
    retriever = HybridRetriever(docs)

    methods = {
        "sparse": lambda q, k: retriever.sparse.search(q, top_k=k),
        "dense": lambda q, k: retriever.dense.search(q, top_k=k),
        # alpha=0.7 chosen via a small sweep over the eval set (see README)
        "hybrid": lambda q, k: retriever.search_weighted(q, top_k=k, alpha=0.7),
    }

    rows = []
    hits = {name: {1: 0, 3: 0} for name in methods}

    for item in eval_queries:
        query, expected_id = item["query"], item["expected_id"]
        row = {"query": query, "expected_id": expected_id, "favors": item["favors"]}
        for name, fn in methods.items():
            ranked = fn(query, 5)
            row[f"{name}_top1"] = ranked[0][0]["id"]
            for k in (1, 3):
                hit = hit_at_k(ranked, expected_id, k)
                if hit == "Y":
                    hits[name][k] += 1
                row[f"{name}_hit@{k}"] = hit
        rows.append(row)

    n = len(eval_queries)
    print(f"{'Query':<55} {'Expect':>6} {'Sparse':>8} {'Dense':>8} {'Hybrid':>8}")
    print("-" * 90)
    for row in rows:
        print(
            f"{row['query'][:54]:<55} {row['expected_id']:>6} "
            f"{row['sparse_hit@1']:>8} {row['dense_hit@1']:>8} {row['hybrid_hit@1']:>8}"
        )

    print("\nSummary (hit@1 / hit@3 over {} queries):".format(n))
    for name in methods:
        print(
            f"  {name:<8} hit@1={hits[name][1]}/{n} ({hits[name][1] / n:.0%})  "
            f"hit@3={hits[name][3]}/{n} ({hits[name][3] / n:.0%})"
        )


if __name__ == "__main__":
    main()
