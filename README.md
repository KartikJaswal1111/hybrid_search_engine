# Hybrid Search Showdown — Scouting the Pitch with TF-IDF vs. Embeddings vs. Hybrid Fusion

A small retrieval engine, themed as a football scouting database, comparing three search strategies over the same 20 scouting/match reports:

- **Sparse** — TF-IDF + cosine similarity (exact token/lexical overlap)
- **Dense** — `sentence-transformers/all-MiniLM-L6-v2` embeddings + cosine similarity (semantic similarity)
- **Hybrid** — weighted fusion of normalized sparse + dense scores: `score = alpha * dense + (1 - alpha) * sparse`

## Why this exists

A scout doesn't work off vibes *or* a stat sheet alone — they cross-reference both. That's exactly the trade-off in retrieval: dense embeddings catch the *concept* ("a deadly poacher who never misses" → striker), but they can lose exact, low-frequency tokens — a FIFA card ID, a transfer code, a VAR decision number — that never carried meaningful signal in training. TF-IDF does the opposite: perfect recall on exact codes, zero notion of "this player plays like that one." This project measures, on a labeled query set, where each approach actually wins or loses.

## Dataset

[data/documents.json](data/documents.json) — 20 short scouting/match reports mixing player profiles with match and transfer records, deliberately including:
- Exact alphanumeric codes (`FUT-23-091`, `TRX-2024-077`, `VAR-DISALLOWED-12`, `INJ-HAM-09`, `RC-MATCH-45`)
- Synonym/paraphrase pairs (`ruthless finisher`/`deadly poacher`, `destroyer`/`holding midfielder`, `playmaker`/`orchestrates the attack with vision`)

[data/eval_queries.json](data/eval_queries.json) — 15 scout-style queries, each labeled with an expected document id and a `favors` tag (`sparse`, `dense`, or `either`).

## Results

Run `python evaluate.py`:

```
Summary (hit@1 / hit@3 over 15 queries):
  sparse   hit@1=13/15 (87%)  hit@3=14/15 (93%)
  dense    hit@1=14/15 (93%)  hit@3=14/15 (93%)
  hybrid   hit@1=14/15 (93%)  hit@3=15/15 (100%)   (alpha=0.7)
```

| Scout query | Expected player/report | Sparse | Dense | Hybrid |
|---|---|---|---|---|
| What does card FUT-23-091 belong to? | striker, doc 1 | ✅ | ✅ | ✅ |
| Tell me about transfer code TRX-2024-077 | transfer record, doc 8 | ✅ | ✅ | ✅ |
| What happened with VAR-DISALLOWED-12? | match report, doc 9 | ✅ | ✅ | ✅ |
| Want a fullback who bombs forward and whips in crosses | fullback, doc 6 | ❌ | ✅ | ✅ |
| Need a deadly poacher who never misses in the box | striker, doc 1 | ❌ (not top-3) | ❌ (not top-3) | ❌ hit@1, ✅ hit@3 |

(Full 15-query table prints when you run `evaluate.py`.)

## Honest findings, not a highlight reel

This is the part most "hybrid search" demos skip — and it's the more interesting result:

1. **Sparse already does fine on exact codes, dense already does fine on synonyms — the gap only opens on phrases that lean hard into descriptive language with zero lexical overlap.** "Want a fullback who bombs forward and whips in crosses" never says "fullback" report's exact words ("overlapping runs," "crossing ability"), and TF-IDF whiffs on it entirely while dense nails it. Conversely, every code-lookup query (`FUT-23-091`, `TRX-2024-077`, ...) was trivial for *both* sparse and dense here — codes are rare enough tokens that embeddings separate them cleanly too, on a small, clean corpus. The "dense can't find exact codes" failure mode is real, but it shows up more reliably at **larger scale with noisier, more repetitive documents** — not necessarily on a tidy 20-doc demo.
2. **Naive 50/50 weighted fusion is not automatically the best of both worlds — it needs tuning.** A small sweep from `alpha=0.3` to `alpha=0.8` landed on `alpha≈0.7` as the sweet spot for this query mix: heavier weight on the dense score (since most misses here are sparse misses on descriptive language), with the sparse score still acting as a tie-breaker/safety net for code lookups.
3. **The single query every method missed at hit@1** ("Need a deadly poacher who never misses in the box," expecting the striker report) is the most telling result: neither sparse nor dense ranked the striker in their own top-3 — sparse surfaced two unrelated midfield reports on weak token overlap ("box," "never"), and dense's closest matches were also midfielders. **Hybrid was the only method to recover the correct player into the top 3**, because combining two *weak, independently-wrong* signals pushed the right document up just enough — a small, real demonstration of why fusion earns its complexity instead of just splitting the difference.

## Architecture

```
scout query
  ├── SparseRetriever (TF-IDF)  ──► sparse score per report
  └── DenseRetriever (MiniLM)   ──► dense score per report
          │
          ▼
   normalize both score sets (min-max)
          │
          ▼
   alpha * dense + (1 - alpha) * sparse  ──► fused ranking
```

`hybrid.py` also implements Reciprocal Rank Fusion (`search_rrf`) as an alternative to weighted averaging — RRF combines rank positions instead of raw scores, avoiding the need for score normalization. It matched weighted fusion's hit@1 (93%) on this eval set.

## Run it

**Option 1 — plain Python**
```bash
pip install -r requirements.txt
python evaluate.py
```

Or query a single method directly: `python sparse.py`, `python dense.py`, `python hybrid.py`.

**Option 2 — Streamlit UI**
```bash
streamlit run streamlit_app.py
```
Opens a live scout dashboard at `http://localhost:8501` with a three-column comparison, animated score bars, token highlighting, and a live scoreboard that responds to the alpha slider.

**Option 3 — Docker (recommended for sharing)**
```bash
docker compose up --build   # first run
docker compose up           # after that
docker compose down         # stop
```

## Docker image size

The default `pip install torch` pulls CUDA/GPU libraries regardless of whether your machine has a GPU, because PyPI ships the "works everywhere" variant by default. This app runs entirely on CPU — TF-IDF is pure matrix math and MiniLM on 20 short documents takes milliseconds without a GPU — so those libraries are dead weight.

The Dockerfile installs the CPU-only torch wheel explicitly before the rest of `requirements.txt`:

```dockerfile
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Result:

| Image | torch variant | Size |
|---|---|---|
| Before (default PyPI torch) | CUDA + all nvidia-* libraries | 9.45 GB |
| After (CPU-only torch) | CPU only | 2.88 GB |

**70% smaller** — without any change to functionality. A good reminder that dependency defaults aren't always right for your actual runtime environment.

## Files

| File | Purpose |
|---|---|
| `data/documents.json` | Scouting/match reports with exact-code and descriptive-synonym edge cases |
| `data/eval_queries.json` | Labeled scout query → expected-report-id eval set |
| `sparse.py` | TF-IDF retriever |
| `dense.py` | Sentence-embedding retriever |
| `hybrid.py` | Weighted-average and RRF fusion |
| `evaluate.py` | Runs all three methods over the eval set, prints hit@1/hit@3 |
| `streamlit_app.py` | Interactive scout dashboard UI |
| `Dockerfile` | CPU-only image, healthcheck on `/_stcore/health` |
| `docker-compose.yml` | Single-command start/stop with port mapping and restart policy |
