"""Scout's Search Dashboard — Google-style clean UI for the hybrid retrieval demo."""

import json
import re
import textwrap

import streamlit as st

from hybrid import HybridRetriever
from sparse import load_documents


def html(content: str) -> None:
    """Render HTML safely — strips leading whitespace from every line so Markdown
    never misreads indented lines as code blocks (4-space rule)."""
    flat = "\n".join(line.strip() for line in textwrap.dedent(content).splitlines())
    st.markdown(flat, unsafe_allow_html=True)


st.set_page_config(page_title="Scout Search", layout="wide", page_icon=None)

st.markdown("""
<style>
section[data-testid="stMain"] > div { background:#ffffff; }
.stApp { background:#ffffff; }
section[data-testid="stSidebar"] { background:#F8F9FA; border-right:1px solid #E8EAED; }
div[data-testid="stTextInput"] input {
    border-radius:24px !important;
    border:1px solid #DFE1E5 !important;
    padding:0.75rem 1.2rem !important;
    font-size:1rem !important;
    box-shadow:0 1px 6px rgba(32,33,36,0.1) !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#4285F4 !important;
    box-shadow:0 1px 12px rgba(66,133,244,0.2) !important;
}
button[kind="secondary"] {
    border-radius:20px !important;
    border:1px solid #DFE1E5 !important;
    background:#F8F9FA !important;
    color:#3C4043 !important;
    font-size:0.8rem !important;
}
button[kind="secondary"]:hover { background:#E8EAED !important; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading scouting models...")
def get_retriever():
    return HybridRetriever(load_documents())


@st.cache_data
def get_eval_queries():
    with open("data/eval_queries.json", encoding="utf-8") as f:
        return json.load(f)


def score_pct(score):
    return max(0.0, min(100.0, float(score) * 100))


def highlight_tokens(query, text):
    tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2}
    return re.sub(
        r"[a-zA-Z0-9]+",
        lambda m: f'<span style="background:#FEF08A;border-radius:2px;padding:0 2px">{m.group(0)}</span>'
        if m.group(0).lower() in tokens else m.group(0),
        text,
    )


def card(rank, doc, score, accent, query, highlight, recovered=False):
    pct      = score_pct(score)
    rank_bg  = "#FBBC05" if rank == 1 else "#F1F3F4"
    rank_col = "#202124" if rank == 1 else "#5F6368"
    text     = highlight_tokens(query, doc["text"]) if highlight else doc["text"]
    badge    = (
        '<div style="display:inline-block;margin-top:8px;font-size:0.68rem;font-weight:600;'
        'color:#137333;background:#E6F4EA;border:1px solid #CEEAD6;border-radius:20px;'
        'padding:2px 10px">Recovered by fusion</div>'
    ) if recovered else ""

    html(f"""
        <div style="background:#fff;border:1px solid #E8EAED;border-radius:12px;
        overflow:hidden;margin-bottom:12px;box-shadow:0 1px 3px rgba(32,33,36,0.08)">
        <div style="height:4px;background:{accent};width:100%"></div>
        <div style="padding:12px 14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <span style="font-size:0.72rem;font-weight:600;background:{rank_bg};color:{rank_col};
        border-radius:20px;padding:2px 10px">#{rank} · Report {doc['id']}</span>
        <span style="font-size:0.82rem;font-weight:700;color:#5F6368">{pct:.0f}%</span>
        </div>
        <div style="height:4px;background:#F1F3F4;border-radius:2px;margin-bottom:10px;overflow:hidden">
        <div style="height:100%;width:{pct:.0f}%;background:{accent};border-radius:2px"></div>
        </div>
        <div style="font-size:0.87rem;color:#3C4043;line-height:1.6">{text}</div>
        {badge}
        </div>
        </div>
    """)


def scoreboard(alpha, eval_queries, retriever):
    n = len(eval_queries)
    hits = {"Keyword": 0, "Semantic": 0, "Hybrid": 0}
    for item in eval_queries:
        q, exp = item["query"], item["expected_id"]
        if retriever.sparse.search(q, top_k=1)[0][0]["id"] == exp:
            hits["Keyword"] += 1
        if retriever.dense.search(q, top_k=1)[0][0]["id"] == exp:
            hits["Semantic"] += 1
        if retriever.search_weighted(q, top_k=1, alpha=alpha)[0][0]["id"] == exp:
            hits["Hybrid"] += 1

    meta = {
        "Keyword":  ("#4285F4", "TF-IDF"),
        "Semantic": ("#EA4335", "Embeddings"),
        "Hybrid":   ("#34A853", f"Fusion a={alpha:.2f}"),
    }
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
        f'<div style="width:90px;font-size:0.82rem;font-weight:500;color:#3C4043">{label}</div>'
        f'<div style="flex:1;height:10px;background:#F1F3F4;border-radius:5px;overflow:hidden">'
        f'<div style="height:100%;width:{hits[name]/n*100:.0f}%;background:{color};border-radius:5px"></div>'
        f'</div>'
        f'<div style="width:40px;text-align:right;font-size:0.82rem;font-weight:700;color:#202124">'
        f'{hits[name]/n*100:.0f}%</div>'
        f'</div>'
        for name, (color, label) in meta.items()
    )

    html(f"""
        <div style="background:#fff;border:1px solid #E8EAED;border-radius:12px;
        padding:16px 18px;margin-top:8px">
        <div style="font-size:0.95rem;font-weight:600;color:#202124;margin-bottom:4px">
        Live scoreboard — hit@1 across 15 labeled queries</div>
        <div style="font-size:0.78rem;color:#5F6368;margin-bottom:14px">
        Drag the fusion weight in the sidebar to watch Hybrid react.</div>
        {rows}
        </div>
    """)


def main():
    retriever    = get_retriever()
    eval_queries = get_eval_queries()

    if "query" not in st.session_state:
        st.session_state.query = "Need a poacher who never misses in the box"

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("**Fusion weight (alpha)**")
        alpha = st.slider("alpha", 0.0, 1.0, 0.7, 0.05, label_visibility="collapsed")
        st.caption("0.0 = pure keyword  ·  1.0 = pure semantic")
        st.divider()
        st.markdown("**Try a preset query**")
        presets = [
            "What does card FUT-23-091 belong to?",
            "Need a poacher who never misses in the box",
            "Looking for a holding midfielder who breaks up play",
            "Want a fullback who bombs forward and whips in crosses",
            "Who's the best penalty taker under pressure?",
        ]
        for p in presets:
            if st.button(p, use_container_width=True):
                st.session_state.query = p
        st.divider()
        with st.expander("View all 20 reports"):
            for d in retriever.documents:
                st.markdown(f"**{d['id']}.** {d['text']}")

    # ── Hero ──
    html("""
        <div style="text-align:center;padding:2.5rem 1rem 1rem">
        <h1 style="font-size:2.8rem;font-weight:700;letter-spacing:-1px;margin:0">
        <span style="color:#4285F4">S</span><span style="color:#EA4335">c</span>
        <span style="color:#FBBC05">o</span><span style="color:#4285F4">u</span>
        <span style="color:#34A853">t</span>&nbsp;Search
        </h1>
        <p style="color:#5F6368;font-size:1rem;margin:0.5rem 0 0">
        Keyword &nbsp;·&nbsp; Semantic &nbsp;·&nbsp; Hybrid — three ways to find the right player report
        </p>
        </div>
    """)

    # ── Search input ──
    _, center, _ = st.columns([1, 2, 1])
    with center:
        query = st.text_input(
            "Scout query",
            key="query",
            placeholder="Search scouting reports...",
            label_visibility="collapsed",
        )

    if not query.strip():
        return

    # ── Results ──
    sparse_r = retriever.sparse.search(query, top_k=3)
    dense_r  = retriever.dense.search(query, top_k=3)
    hybrid_r = retriever.search_weighted(query, top_k=3, alpha=alpha)

    sparse_ids = {d["id"] for d, _ in sparse_r}
    dense_ids  = {d["id"] for d, _ in dense_r}

    col1, col2, col3 = st.columns(3)

    with col1:
        html('<div style="display:inline-block;background:#E8F0FE;color:#1A73E8;'
             'font-size:0.82rem;font-weight:600;padding:5px 14px;border-radius:20px;'
             'margin-bottom:10px">Keyword (TF-IDF)</div>')
        for rank, (doc, score) in enumerate(sparse_r, 1):
            card(rank, doc, score, "#4285F4", query, highlight=True)

    with col2:
        html('<div style="display:inline-block;background:#FCE8E6;color:#D93025;'
             'font-size:0.82rem;font-weight:600;padding:5px 14px;border-radius:20px;'
             'margin-bottom:10px">Semantic (Embeddings)</div>')
        for rank, (doc, score) in enumerate(dense_r, 1):
            card(rank, doc, score, "#EA4335", query, highlight=False)

    with col3:
        html('<div style="display:inline-block;background:#E6F4EA;color:#137333;'
             'font-size:0.82rem;font-weight:600;padding:5px 14px;border-radius:20px;'
             'margin-bottom:10px">Hybrid Fusion</div>')
        for rank, (doc, score) in enumerate(hybrid_r, 1):
            recovered = doc["id"] not in sparse_ids and doc["id"] not in dense_ids
            card(rank, doc, score, "#34A853", query, highlight=True, recovered=recovered)

    # ── Scoreboard ──
    html('<hr style="border:none;border-top:1px solid #E8EAED;margin:1.5rem 0 1rem">')
    scoreboard(alpha, eval_queries, retriever)


if __name__ == "__main__":
    main()
