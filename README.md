# Hybrid Search Engine: Scouting the Pitch with TF-IDF vs. Embeddings vs. Hybrid Fusion

A production-minded retrieval engine, themed as a soccer scouting database, comparing three search strategies on the same 20 scouting and match reports — measured on a labeled eval set, containerised with Docker, and served through an interactive UI.

- **Sparse** — TF-IDF + cosine similarity (exact token/lexical overlap)
- **Dense** — `sentence-transformers/all-MiniLM-L6-v2` embeddings + cosine similarity (semantic similarity)
- **Hybrid** — weighted fusion: `score = alpha * dense + (1 - alpha) * sparse`, tuned via eval sweep.

---

## The Business Problem This Solves

Every product with a search bar faces the same split:

| Customer behaviour | Example | Best method |
|---|---|---|
| Pastes an exact ID or code | "Order #TRX-2024-077", "Error code ERR-99X" | Sparse (TF-IDF) |
| Describes what they want | "A blue jacket that doesn't look bulky" | Dense (Embeddings) |
| Mix of both | Most real users, most of the time | Hybrid |

Most teams pick one method, usually embeddings because it's trendy, and quietly eat the failure cases on the other side. The result: customers searching for exact SKUs get irrelevant semantic matches, and customers describing a concept get zero results because the keyword didn't appear verbatim.

This project measures exactly where each method breaks, demonstrates the failure cases with real labeled queries, and shows how a weighted fusion layer recovers the correct answer in cases where both methods individually failed.

**Real-world domains where this pattern appears:**
- **E-commerce** — exact SKU / product code lookups vs. natural language product descriptions
- **Customer support** — exact error code search vs. "my app keeps crashing" symptom descriptions
- **Legal / compliance** — exact clause number lookup vs. conceptual policy questions
- **Healthcare** — exact drug code vs. symptom-based queries
- **Soccer scouting** — exact card ID or transfer code vs. "need a destroyer who breaks up play"

---

## Why this exists

A scout doesn't work off vibes *or* a stat sheet alone — they cross-reference both. Dense embeddings catch the *concept* ("a poacher who never misses" → striker), but can lose exact low-frequency tokens — a FIFA card ID, a transfer code, a VAR decision number. TF-IDF does the opposite: perfect recall on exact codes, zero notion of meaning. This project measures, on a labeled query set, where each approach actually wins or loses — and proves that combining two independently wrong signals can produce a right answer.

---

## Results

Run `python evaluate.py`:

```
Summary (hit@1 / hit@3 over 15 queries):
  sparse   hit@1=13/15 (87%)  hit@3=14/15 (93%)
  dense    hit@1=14/15 (93%)  hit@3=14/15 (93%)
  hybrid   hit@1=14/15 (93%)  hit@3=15/15 (100%)   (alpha=0.7)
```

| Scout query | Expected | Sparse | Dense | Hybrid |
|---|---|---|---|---|
| What does card FUT-23-091 belong to? | striker, doc 1 | ✅ | ✅ | ✅ |
| What happened with VAR-DISALLOWED-12? | match report, doc 9 | ✅ | ✅ | ✅ |
| Want a fullback who bombs forward and whips in crosses | fullback, doc 6 | ❌ | ✅ | ✅ |
| Need a poacher who never misses in the box | striker, doc 1 | ❌ not top-3 | ❌ not top-3 | ✅ recovered at #3 |

(Full 15-query table prints when you run `evaluate.py`.)

---

## The Finding That Makes This Interesting

The query `"Need a poacher who never misses in the box"` (expected: striker report) was the most telling result:

```
Sparse top 3:  #1 doc11 [0.329]  Set-piece specialist  ❌
               #2 doc12 [0.225]  Box-to-box midfielder ❌
               #3 doc19 [0.164]  Hat-trick match report ❌

Dense top 3:   #1 doc20 [0.289]  Free-kick specialist  ❌
               #2 doc11 [0.284]  Set-piece specialist  ❌
               #3 doc12 [0.281]  Box-to-box midfielder ❌

Hybrid top 3:  #1 doc11 [0.989]  Set-piece specialist  ❌
               #2 doc12 [0.887]  Box-to-box midfielder ❌
               #3 doc1  [0.802]  Striker report        ✅ recovered
```

Neither sparse nor dense ranked the striker in their own top 3. But the striker had a weak non-zero signal in both — not enough to win alone, but when combined through the fusion layer, the accumulated signal pushed it to #3. **Two independently wrong systems corrected each other.**

---

## Honest Findings, Not a Highlight Reel

1. **Naive 50/50 fusion actively hurts performance.** Starting at `alpha=0.5` scored 80% hit@1 — worse than dense alone (93%). Averaging in a noisy sparse score dragged correct dense answers down. A sweep from `alpha=0.3` to `alpha=0.8` found `alpha=0.7` as the sweet spot. Fusion weighting is a hyperparameter, not a default.

2. **The failure mode is scale-dependent.** On this clean 20-doc corpus, dense handles exact codes well because the tokens are distinctive enough. The "dense misses exact codes" failure shows up more reliably on larger, noisier corpora where many documents share surface-level similarity. The architecture is designed for that scale.

3. **Ground truth is an assumption.** The one query all methods missed at hit@1 ("poacher in the box") is arguably a labeling judgment call — the set-piece specialist (ranked #1 by hybrid) is a defensible answer too. Evaluating retrieval honestly means interrogating your own eval set, not just your retrieval methods.

---

## Architecture

```
scout query
  ├── SparseRetriever (TF-IDF)  ──► sparse score per report
  └── DenseRetriever (MiniLM)   ──► dense score per report
          │
          ▼
   normalize both score sets (min-max per query)
          │
          ▼
   alpha * dense + (1 - alpha) * sparse  ──► fused ranking
```

`hybrid.py` also implements **Reciprocal Rank Fusion (RRF)** as an alternative — combines rank positions instead of raw scores, no normalization needed. Matched weighted fusion's hit@1 (93%) on this eval set.

---

## Streamlit UI

An interactive dashboard that makes the comparison visual:

- Three-column layout: Keyword (Blue) · Semantic (Red) · Hybrid (Green)
- Animated score bars per result card
- Yellow token highlighting on keyword-matched words — visually proves why TF-IDF found or missed a result
- **"Recovered by fusion" badge** — automatically appears when hybrid surfaces a result neither method ranked individually
- **Live scoreboard** — hit@1 bars across all 15 eval queries that recompute in real time as you drag the alpha slider in the sidebar

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## Run it

**Option 1 — plain Python**
```bash
pip install -r requirements.txt
python evaluate.py
```

**Option 2 — Streamlit UI**
```bash
streamlit run streamlit_app.py
```

**Option 3 — Docker (recommended)**
```bash
docker compose up --build   # first run
docker compose up           # every run after
docker compose down         # stop
```

---

## Docker Image Optimisation

The default `pip install torch` pulls every CUDA/GPU library regardless of whether the machine has a GPU — PyPI ships the "works everywhere" variant by default. This app runs entirely on CPU (TF-IDF is pure matrix math, MiniLM on 20 docs takes milliseconds without a GPU).

```dockerfile
# Install CPU-only torch before requirements.txt so pip never pulls CUDA
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

| Image | Torch variant | Size |
|---|---|---|
| Default PyPI torch | CUDA + all nvidia-* libraries | 9.45 GB |
| CPU-only torch | CPU only | 2.88 GB |

**70% smaller, zero change in functionality.** Dependency defaults are not always right for your runtime environment.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Sparse retrieval | scikit-learn TF-IDF |
| Dense retrieval | sentence-transformers MiniLM-L6-v2 |
| Fusion | Weighted average + RRF (hybrid.py) |
| Evaluation | Custom hit@1 / hit@3 harness (evaluate.py) |
| UI | Streamlit (Google-style custom CSS) |
| Containerisation | Docker + Docker Compose |
| Base image | python:3.11-slim (CPU-only) |

---

## Files

| File | Purpose |
|---|---|
| `data/documents.json` | 20 scouting/match reports with exact-code and synonym edge cases |
| `data/eval_queries.json` | 15 labeled scout queries with expected doc ids and favors tags |
| `sparse.py` | TF-IDF retriever |
| `dense.py` | Sentence-embedding retriever |
| `hybrid.py` | Weighted-average and RRF fusion |
| `evaluate.py` | Runs all three methods, prints hit@1/hit@3 comparison table |
| `streamlit_app.py` | Interactive scout dashboard |
| `Dockerfile` | CPU-only optimised image with healthcheck |
| `docker-compose.yml` | Single-command start/stop with port mapping and restart policy |
| `.streamlit/config.toml` | Light theme |
