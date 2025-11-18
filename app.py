import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Capstone Similarity Checker",
    page_icon="🔎",
    layout="wide",
)
st.title("🔎 Capstone Similarity Checker")
st.caption(
    "Live Google Sheet → Lexical gate (TF-IDF) + Semantic refinement (SBERT) → "
    "Top matches with interpretable similarity labels."
)

# ===================== DEFAULTS =====================
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQOt3ScW1TkCpKVCP2vNMbNSahbMkaFZBARjoTRe267tQdX_E_hC8o3bXTjwkhPxdXKtKfq1_dWLZMU/pub?gid=1929751519&single=true&output=csv"
)
PREFERRED_TITLE_NAMES = [
    "Project Title",
    "Title",
    "Capstone Title",
    "title",
    "project_title",
    "Project",
]

# ===================== TEXT HELPERS =====================
def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def expand_query(q: str) -> str:
    """Small typo fixes + expansions for common abbreviations."""
    q2 = normalize_text(q)
    fixes = {
        "artifical": "artificial",
        "intellegence": "intelligence",
        "machne": "machine",
        "lern": "learning",
    }
    for wrong, right in fixes.items():
        q2 = q2.replace(wrong, right)
    # simple expansions
    if q2 == "ai" or " ai " in f" {q2} ":
        q2 += " artificial intelligence"
    if q2 == "ml" or " ml " in f" {q2} ":
        q2 += " machine learning"
    if "nlp" in q2:
        q2 += " natural language processing"
    return q2


def strength_label(score: float) -> str:
    """Convert hybrid score in [0,1] to a verbal band."""
    if score < 0.30:
        return "Weak"
    elif score < 0.60:
        return "Moderate"
    elif score < 0.80:
        return "Strong"
    else:
        return "Very strong"


def tokenize_simple(text: str):
    """Simple tokenization for word frequency stats."""
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 2]
    return tokens


# ===================== MODEL LOADERS =====================
@st.cache_resource(show_spinner=False)
def load_sbert_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data(show_spinner=False)
def build_lexical_index(titles: pd.Series):
    """
    Word-level TF-IDF for lexical similarity.
    Uses unigrams + bigrams and English stopwords.
    """
    titles_norm = titles.astype(str).apply(normalize_text)
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        max_df=1.0,
    )
    mat = vec.fit_transform(titles_norm.tolist())
    return vec, mat


@st.cache_data(show_spinner=False)
def embed_titles_sbert(titles: pd.Series):
    model = load_sbert_model()
    titles_norm = titles.astype(str).apply(normalize_text).tolist()
    embs = model.encode(
        titles_norm, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(embs)


# ===================== DATA HELPERS =====================
@st.cache_data(show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


@st.cache_data(show_spinner=False)
def detect_title_column(df: pd.DataFrame) -> str:
    for c in PREFERRED_TITLE_NAMES:
        if c in df.columns:
            return c
    # fallback: first text-like column
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            return c
    raise ValueError("No suitable text column found for titles.")


@st.cache_data(show_spinner=False)
def clean_titles(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("").str.strip()
    s = s[s.str.len() > 0].reset_index(drop=True)
    return s


# ===================== CORE SIMILARITY ENGINE =====================
def find_similar_titles(
    query: str,
    titles: pd.Series,
    lex_vec,
    lex_mat,
    sbert_embs,
    top_k: int = 5,
    lex_gate_strong: float = 0.15,   # strong lexical match threshold
    lex_gate_candidate: float = 0.10,  # min lexical sim to be considered candidate
    w_semantic: float = 0.70,
    w_lexical: float = 0.30,
):
    """
    Lexical gate + semantic refinement.

    Returns:
      results_df: DataFrame with matches and scores
      info: dict with summary flags / messages
    """
    q = expand_query(query)
    if not q:
        return None, {"error": "Please enter a non-empty title."}

    # ---------- LEXICAL SIMILARITY (WORD TF-IDF) ----------
    q_vec = lex_vec.transform([q])
    lex_sim = cosine_similarity(q_vec, lex_mat).ravel()  # in [0,1]
    max_lex = float(lex_sim.max())

    strong_lexical_match = max_lex >= lex_gate_strong

    # Candidates: titles with at least some lexical overlap
    cand_mask = lex_sim >= lex_gate_candidate
    cand_idx = np.where(cand_mask)[0]

    info = {
        "max_lexical_sim": max_lex,
        "strong_lexical_match": strong_lexical_match,
        "num_candidates": int(len(cand_idx)),
    }

    if len(cand_idx) == 0:
        info["message"] = (
            "No titles share enough wording with the query. "
            "We consider this title lexically novel in this dataset."
        )
        empty = pd.DataFrame(
            columns=["Existing Title", "Lexical", "Semantic", "Hybrid", "Hybrid %", "Strength"]
        )
        return empty, info

    # ---------- SEMANTIC SIMILARITY (SBERT) ON CANDIDATES ----------
    model = load_sbert_model()
    q_emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    cand_sbert = sbert_embs[cand_idx]                       # embeddings for candidates
    sem_sim_raw = cand_sbert @ q_emb                        # cosine in [-1,1]
    sem_sim = np.clip((sem_sim_raw + 1.0) / 2.0, 0.0, 1.0)  # -> [0,1]

    cand_lex = lex_sim[cand_idx]                            # lexical sims for candidates

    # ---------- HYBRID SCORE ----------
    hybrid = (
        w_semantic * sem_sim +
        w_lexical * cand_lex
    )

    # ---------- RANK & FORMAT RESULTS ----------
    order = np.argsort(-hybrid)
    k = min(top_k, len(order))
    sel = order[:k]
    idx_sel = cand_idx[sel]

    results = pd.DataFrame({
        "Existing Title": titles.iloc[idx_sel].values,
        "Lexical": np.round(cand_lex[sel], 4),
        "Semantic": np.round(sem_sim[sel], 4),
        "Hybrid": np.round(hybrid[sel], 4),
    })
    results["Hybrid %"] = (results["Hybrid"] * 100).round(2)
    results["Strength"] = results["Hybrid"].apply(strength_label)

    if not strong_lexical_match:
        info["message"] = (
            "Only weak lexical overlaps found. These are the closest titles in the dataset, "
            "but the new title is likely a novel topic."
        )
    else:
        info["message"] = (
            "At least one existing title shares meaningful wording with the query. "
            "Review matches labelled 'Strong' or 'Very strong' for potential overlap."
        )

    return results, info


# ===================== UI: DATA SOURCE =====================
with st.expander("📄 Data Source (Google Sheet CSV)"):
    sheet_url = st.text_input(
        "Paste your **published-to-web CSV** link (must end with `output=csv`):",
        value=DEFAULT_SHEET_URL,
    )
    st.caption("Google Sheets → File → Share → Publish to web → CSV → copy the link.")

df = None
err_box = st.empty()
try:
    df = fetch_csv(sheet_url)
except Exception as e:
    err_box.error(
        "Could not load CSV. Ensure the link is public & ends with `output=csv`.\n\n"
        f"Error: {e}"
    )

if df is not None:
    try:
        title_col = detect_title_column(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    titles = clean_titles(df[title_col])
    if titles.empty:
        st.error("No non-empty titles found in the detected column.")
        st.stop()

    # Precompute indices once (cached)
    lex_vec, lex_mat = build_lexical_index(titles)
    sbert_embs = embed_titles_sbert(titles)

    # Try to detect a year column for insights
    year_col = None
    for cand in ["Year", "year", "Academic Year", "Year_of_Completion"]:
        if cand in df.columns:
            year_col = cand
            break

    # ===================== TABS =====================
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Similarity Checker", "📊 Dataset Insights", "ℹ️ Model Info"]
    )

    # ---------- TAB 1: SIMILARITY CHECKER ----------
    with tab1:
        st.markdown("### Check a new capstone title")
        qcol1, qcol2 = st.columns([2, 1])
        with qcol1:
            query_title = st.text_input(
                "Enter a new capstone title to check:",
                placeholder="e.g., AI-based demand forecasting for retail supply chains",
                key="query_title_main",
            )
        with qcol2:
            top_k = st.number_input(
                "Top matches",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                key="top_k_main",
            )

        if st.button("Check Similarity", type="primary", use_container_width=True):
            results_df, info = find_similar_titles(
                query_title,
                titles,
                lex_vec,
                lex_mat,
                sbert_embs,
                top_k=int(top_k),
            )

            if "error" in info:
                st.warning(info["error"])
            else:
                st.markdown(f"**Summary:** {info['message']}")
                st.caption(
                    f"Max lexical similarity in dataset: {info['max_lexical_sim']:.3f} "
                    f"(candidates after gate: {info['num_candidates']})"
                )

                if results_df.empty:
                    st.info("No candidate titles to display.")
                else:
                    st.subheader("Top candidate matches")
                    st.dataframe(results_df, use_container_width=True)

                    st.download_button(
                        "⬇️ Download results (CSV)",
                        data=results_df.to_csv(index=False).encode("utf-8"),
                        file_name="similarity_results.csv",
                        mime="text/csv",
                    )

    # ---------- TAB 2: DATASET INSIGHTS ----------
    with tab2:
        st.markdown("### Dataset overview and trends")

        # Basic stats
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total projects", len(titles))
        col_b.metric("Unique titles", titles.nunique())
        avg_len = titles.str.len().mean()
        col_c.metric("Avg. title length (chars)", f"{avg_len:.1f}")

        st.markdown("#### Year-wise project distribution")
        if year_col:
            year_counts = df[year_col].value_counts().sort_index()
            st.bar_chart(year_counts, use_container_width=True)
            st.caption(f"Projects per {year_col}")
        else:
            st.info(
                "No 'Year' column detected. "
                "Add a year column in your Google Sheet to see temporal trends."
            )

        st.markdown("#### Title length distribution (in words)")
        title_word_counts = titles.apply(lambda t: len(str(t).split()))
        length_counts = (
            title_word_counts.value_counts().sort_index().rename("Count")
        )
        st.bar_chart(length_counts, use_container_width=True)
        st.caption("Most titles are short; this explains why we need a robust hybrid model.")

        # Keyword frequency
        st.markdown("#### Most common keywords in titles")
        all_tokens = []
        for t in titles:
            all_tokens.extend(tokenize_simple(t))

        # Very small stopword list (you can extend later)
        manual_stopwords = {
            "the", "and", "for", "with", "from", "into",
            "case", "study", "using", "based", "city",
            "cities", "urban", "plan", "planning",
        }
        freq = {}
        for tok in all_tokens:
            if tok in manual_stopwords:
                continue
            freq[tok] = freq.get(tok, 0) + 1

        if freq:
            freq_df = (
                pd.DataFrame(
                    [{"keyword": k, "count": v} for k, v in freq.items()]
                )
                .sort_values("count", ascending=False)
                .head(20)
                .set_index("keyword")
            )
            st.bar_chart(freq_df, use_container_width=True)
            st.caption("Top 20 non-trivial keywords across project titles.")
        else:
            st.info("Not enough textual data to compute keyword frequencies.")

        # Simple theme tagging (very rough, for visual flavour)
        st.markdown("#### Approximate thematic distribution")
        def classify_theme(title: str) -> str:
            t = normalize_text(title)
            if any(k in t for k in ["ai", "artificial intelligence", "machine learning", "data", "analytics"]):
                return "AI / Data / Analytics"
            if any(k in t for k in ["transport", "mobility", "bus", "rail", "transit", "traffic"]):
                return "Transport / Mobility"
            if any(k in t for k in ["housing", "homeless", "rent", "affordability", "gentrification"]):
                return "Housing / Urban Equity"
            if any(k in t for k in ["health", "hospital", "clinic", "mental", "care"]):
                return "Health / Healthcare"
            if any(k in t for k in ["climate", "environment", "sustainab", "green"]):
                return "Environment / Climate"
            if any(k in t for k in ["policy", "governance", "regulation"]):
                return "Policy / Governance"
            return "Other / Mixed"

        theme_series = titles.apply(classify_theme)
        theme_counts = theme_series.value_counts().sort_values(ascending=False)
        st.bar_chart(theme_counts, use_container_width=True)
        st.caption(
            "Rough theme classification based on simple keyword rules. "
            "This gives faculty a quick view of popular focus areas."
        )

        with st.expander("Preview (first 10 titles)"):
            st.write(pd.DataFrame(titles.head(10), columns=["Title"]))

    # ---------- TAB 3: MODEL INFO ----------
    with tab3:
        st.markdown("### How the similarity model works")
        st.markdown(
            """
            **High-level pipeline**

            1. **Load data** from a live Google Sheet (published as CSV).
            2. **Preprocess titles** (lowercasing, trimming, small typo fixes for the query).
            3. **Lexical similarity (TF-IDF + cosine):**
               - Build a word-level TF-IDF index over all titles (unigrams + bigrams, English stopwords removed).
               - For a new title, compute cosine similarity with all existing titles → **Lexical score in [0,1]**.
               - Filter out titles with negligible lexical overlap (lexical < 0.10).
            4. **Semantic similarity (SBERT embeddings):**
               - Use `sentence-transformers/all-MiniLM-L6-v2` to encode titles and the query.
               - Cosine similarity between embeddings (mapped from [-1,1] to [0,1]) → **Semantic score**.
            5. **Hybrid score:**
               - For candidate titles, compute  
                 `Hybrid = 0.7 × Semantic + 0.3 × Lexical`.
               - This balances genuine meaning with actual word overlap.
            6. **Ranking & labels:**
               - Sort by Hybrid score.
               - Convert to percentage and classify as **Weak / Moderate / Strong / Very strong**.

            **Why hybrid?**

            - Titles are **short**, so purely semantic models can behave unpredictably.
            - TF-IDF ensures we maintain **grounding in actual vocabulary**.
            - SBERT adds robustness to paraphrasing and synonyms.
            - The 0.7 / 0.3 weighting is based on empirical tuning on this dataset:
              semantic signal is more informative, but lexical overlap must still matter.

            **Intended use**

            - The tool is designed as a **decision-support assistant**, not an automatic judge.
            - It helps students and faculty quickly inspect potential overlaps and encourages more original project ideas.
            """
        )

st.markdown("---")
st.caption(
    "Tip: Use the Similarity Checker tab for new titles, and the Dataset Insights tab to understand trends in past capstone topics."
)
