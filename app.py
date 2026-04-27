"""
Intent Send Generator
---------------------
One-page Streamlit app that turns a Biscred contact pool + a weekly Bombora
intent file into a "send" CSV (max-N per firm, total target). Tracks sent
contacts in a local SQLite file so they're never sent twice.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import random
import re
import sqlite3
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).parent / "data" / "intent.db"
DB_PATH.parent.mkdir(exist_ok=True)

CONTACT_COLS = [
    "uuid", "contact_name", "job_title", "job_classifications", "company_name",
    "comp_industries", "comp_asset_classes", "website", "contact_email",
    "linkedin_url", "full_phone_number", "locality", "region",
]
OUTPUT_COLS = CONTACT_COLS + ["Average Score", "Topic Count"]
COMPANY_SUFFIXES = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited", "corp",
    "corporation", "co", "company", "plc", "lp", "llp", "lllp", "gmbh",
    "ag", "sa", "nv", "bv", "holdings", "group",
}


# ---------- DB ----------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contacts (
            uuid TEXT PRIMARY KEY,
            contact_name TEXT,
            job_title TEXT,
            job_classifications TEXT,
            company_name TEXT,
            comp_industries TEXT,
            comp_asset_classes TEXT,
            website TEXT,
            contact_email TEXT,
            linkedin_url TEXT,
            full_phone_number TEXT,
            locality TEXT,
            region TEXT,
            domain_norm TEXT,
            company_norm TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sent (
            uuid TEXT PRIMARY KEY,
            sent_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contacts_domain ON contacts(domain_norm)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_contacts_company ON contacts(company_norm)")
    conn.commit()
    return conn


# ---------- Normalization ----------
def normalize_domain(s) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    if not s:
        return ""
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    s = s.split("/")[0].split("?")[0].split("#")[0]
    return s.strip()


def normalize_company(s) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t and t not in COMPANY_SUFFIXES]
    return " ".join(tokens)


def read_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    # Prefer the first matching column, and skip columns that are entirely empty
    # (Bombora often has duplicate-named columns like "Company Name" and "company name").
    cands = {c.lower() for c in candidates}
    matches = [c for c in df.columns if c.lower().strip() in cands]
    if not matches:
        return None
    for c in matches:
        if df[c].notna().any():
            return c
    return matches[0]


# ---------- Selection ----------
def generate_send(
    conn: sqlite3.Connection,
    bombora: pd.DataFrame,
    total_target: int,
    max_per_firm: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    domain_col = find_col(bombora, "Domain")
    score_col = find_col(bombora, "Average Score", "Avg Score", "Score")
    topic_col = find_col(bombora, "Topic Count", "Topics")
    name_col = find_col(bombora, "Company Name", "company_name")

    if not domain_col or not score_col:
        raise ValueError("Bombora file must contain at least 'Domain' and 'Average Score' columns.")

    bom = bombora.copy()
    bom["_domain"] = bom[domain_col].apply(normalize_domain)
    bom["_company"] = bom[name_col].apply(normalize_company) if name_col else ""
    sort_cols = [score_col] + ([topic_col] if topic_col else [])
    bom = bom.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)

    sent_uuids = {row[0] for row in conn.execute("SELECT uuid FROM sent")}
    rng = random.Random(seed)

    picked: list[list] = []
    used_uuids: set[str] = set()
    matched_firms = 0
    skipped_no_match: list[str] = []

    placeholder_select = (
        "SELECT uuid, contact_name, job_title, job_classifications, company_name, "
        "comp_industries, comp_asset_classes, website, contact_email, linkedin_url, "
        "full_phone_number, locality, region "
    )

    for _, frow in bom.iterrows():
        if len(picked) >= total_target:
            break
        domain = frow["_domain"]
        company = frow["_company"]
        rows: list = []
        if domain:
            rows = conn.execute(
                placeholder_select + "FROM contacts WHERE domain_norm = ?",
                (domain,),
            ).fetchall()
        if not rows and company:
            rows = conn.execute(
                placeholder_select + "FROM contacts WHERE company_norm = ?",
                (company,),
            ).fetchall()
        available = [r for r in rows if r[0] not in sent_uuids and r[0] not in used_uuids]
        if not available:
            label = (str(frow.get(name_col)) if name_col else "") or domain or company or "?"
            skipped_no_match.append(label)
            continue
        matched_firms += 1
        n = min(max_per_firm, len(available), total_target - len(picked))
        chosen = rng.sample(available, n)
        avg_score = frow[score_col]
        topic_count = frow[topic_col] if topic_col else None
        for r in chosen:
            used_uuids.add(r[0])
            picked.append(list(r) + [avg_score, topic_count])

    result = pd.DataFrame(picked, columns=OUTPUT_COLS)
    stats = {
        "rows": len(result),
        "firms_in_bombora": len(bom),
        "firms_matched": matched_firms,
        "firms_skipped": len(skipped_no_match),
        "skipped_examples": skipped_no_match[:10],
    }
    return result, stats


BISCRED_EXPORT_ALIASES = {
    "Job Title": "job_title",
    "Asset Experience": "job_classifications",
    "Company": "company_name",
    "Company Industries": "comp_industries",
    "Company Asset Experience": "comp_asset_classes",
    "Email": "contact_email",
    "LinkedIn": "linkedin_url",
    "Phone Number": "full_phone_number",
    "City": "locality",
    "State": "region",
    "Website": "website",
}


def _domain_from_email(s) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    if "@" not in s:
        return ""
    return s.split("@", 1)[1]


def load_contact_pool(conn: sqlite3.Connection, df: pd.DataFrame, replace: bool) -> int:
    df = df.copy()
    if "uuid" not in df.columns:
        raise ValueError("Contact CSV must have a 'uuid' column.")

    for src, dst in BISCRED_EXPORT_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    if "contact_name" not in df.columns and {"First Name", "Last Name"} <= set(df.columns):
        first = df["First Name"].fillna("").astype(str).str.strip()
        last = df["Last Name"].fillna("").astype(str).str.strip()
        df["contact_name"] = (first + " " + last).str.strip()
    if ("website" not in df.columns or df["website"].isna().all()) and "contact_email" in df.columns:
        df["website"] = df["contact_email"].apply(_domain_from_email)

    for c in CONTACT_COLS:
        if c not in df.columns:
            df[c] = None
    df["domain_norm"] = df["website"].apply(normalize_domain)
    df["company_norm"] = df["company_name"].apply(normalize_company)
    df = df[CONTACT_COLS + ["domain_norm", "company_norm"]]
    df = df.drop_duplicates(subset=["uuid"], keep="last")

    cur = conn.cursor()
    if replace:
        cur.execute("DELETE FROM contacts")
    rows = [tuple(r) for r in df.itertuples(index=False, name=None)]
    cur.executemany(
        f"INSERT OR REPLACE INTO contacts ({', '.join(CONTACT_COLS + ['domain_norm', 'company_norm'])}) "
        f"VALUES ({', '.join(['?'] * (len(CONTACT_COLS) + 2))})",
        rows,
    )
    conn.commit()
    return len(rows)


# ---------- Horn ----------
@st.cache_resource
def _horn_wav_b64() -> str:
    sr = 22050
    notes = [(110.00, 0.55), (164.81, 1.20)]  # A2 → E3, a war-horn fifth
    chunks: list[np.ndarray] = []
    for freq, dur in notes:
        t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
        tone = (
            0.50 * np.sin(2 * np.pi * freq * t)
            + 0.30 * np.sin(2 * np.pi * freq * 2 * t)
            + 0.18 * np.sin(2 * np.pi * freq * 3 * t)
            + 0.08 * np.sin(2 * np.pi * freq * 4 * t)
        )
        env = np.ones_like(t)
        a = int(0.04 * sr)
        r = int(0.22 * sr)
        env[:a] = np.linspace(0.0, 1.0, a)
        env[-r:] = np.linspace(1.0, 0.0, r)
        chunks.append(tone * env)
    audio = np.clip(np.concatenate(chunks), -1.0, 1.0)
    pcm = (audio * 30000).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sound_the_horn() -> None:
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{_horn_wav_b64()}" '
        'type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


# ---------- Scroll rustle ----------
@st.cache_resource
def _scroll_wav_b64() -> str:
    sr = 22050
    dur = 0.55
    n = int(sr * dur)
    rng = np.random.default_rng(11)

    # Soft mid-band rustle: white noise smoothed by a short moving average
    noise = rng.standard_normal(n)
    k = 6
    rustle = np.convolve(noise, np.ones(k) / k, mode="same")

    # A handful of sharper "scratches" to suggest paper unfurling
    for _ in range(7):
        pos = int(rng.integers(0, n - 120))
        ln = int(rng.integers(40, 120))
        decay = np.exp(-np.linspace(0.0, 4.0, ln))
        rustle[pos:pos + ln] += rng.standard_normal(ln) * decay * 0.7

    env = np.ones(n)
    a = int(0.02 * sr)
    r = int(0.30 * sr)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r)

    audio = rustle * env
    audio = audio / max(float(np.max(np.abs(audio))), 1e-9) * 0.5
    pcm = (audio * 30000).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _play_scroll_if_new(uploaded, key: str) -> None:
    if uploaded is None:
        return
    fid = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{getattr(uploaded, 'size', '')}"
    state_key = f"_last_upload_{key}"
    if st.session_state.get(state_key) == fid:
        return
    st.session_state[state_key] = fid
    st.markdown(
        f'<audio autoplay><source src="data:audio/wav;base64,{_scroll_wav_b64()}" '
        'type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


# ---------- Hearth (fire ambience loop) ----------
@st.cache_resource
def _fire_wav_b64() -> str:
    sr = 22050
    duration = 10.0
    L = int(sr * duration)
    xf = int(0.6 * sr)
    total = L + xf
    rng = np.random.default_rng(7)

    white = rng.standard_normal(total)

    # Low-frequency rumble: heavily smoothed white noise
    rumble = np.convolve(white, np.ones(48) / 48, mode="same")
    rumble /= max(float(np.max(np.abs(rumble))), 1e-9)

    # Mid-band hiss for a constant burning bed
    hiss = np.convolve(white, np.ones(8) / 8, mode="same")
    hiss /= max(float(np.max(np.abs(hiss))), 1e-9)

    # Sparse crackle pops with sharp attack and exponential decay
    crackle = np.zeros(total)
    n_pops = int(duration * 5)
    for _ in range(n_pops):
        pos = int(rng.integers(0, total - 400))
        ln = int(rng.integers(40, 320))
        amp = float(rng.uniform(0.25, 0.95))
        decay = np.exp(-np.linspace(0.0, 7.0, ln))
        crackle[pos:pos + ln] += rng.standard_normal(ln) * decay * amp

    audio = rumble * 0.45 + hiss * 0.18 + crackle * 0.55

    # Slow "breathing" amplitude modulation so the fire feels alive
    anchors = rng.standard_normal(int(duration) + 2)
    breath = np.interp(
        np.arange(total),
        np.linspace(0, total - 1, len(anchors)),
        anchors,
    )
    breath /= max(float(np.max(np.abs(breath))), 1e-9)
    audio *= 1.0 + 0.25 * breath

    # Seamless loop: crossfade the leading xf samples with the tail xf samples
    # so output[L-1] meets output[0] at adjacent positions in the source signal.
    fade_in = np.linspace(0.0, 1.0, xf)
    fade_out = 1.0 - fade_in
    out = np.empty(L)
    out[:xf] = audio[:xf] * fade_in + audio[L:L + xf] * fade_out
    out[xf:] = audio[xf:L]

    out = out / max(float(np.max(np.abs(out))), 1e-9) * 0.55
    pcm = (out * 30000).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_hearth(lit: bool) -> None:
    if not lit:
        return
    st.markdown(
        f'<audio autoplay loop><source src="data:audio/wav;base64,{_fire_wav_b64()}" '
        'type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


# ---------- UI ----------
boring_mode = st.session_state.get("boring_mode", False)

st.set_page_config(
    page_title="Intent Send Generator" if boring_mode else "The Intent Forge",
    page_icon="📊" if boring_mode else "⚔",
    layout="wide",
)


def t(fancy: str, boring: str) -> str:
    """Return the boring or fancy variant of a string based on the current mode."""
    return boring if boring_mode else fancy

DAGGERFALL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=IM+Fell+English:ital@0;1&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at top, #2a1f15 0%, #14100a 70%, #0a0705 100%);
    color: #e6d4a8;
}

/* Faint stone-block grain over the whole app */
.stApp::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none;
    background-image:
        repeating-linear-gradient(0deg, rgba(0,0,0,0.06) 0 2px, transparent 2px 4px),
        repeating-linear-gradient(90deg, rgba(0,0,0,0.04) 0 2px, transparent 2px 4px);
    z-index: 0;
}

h1, h2, h3, h4 {
    font-family: 'Cinzel', serif !important;
    color: #d4af37 !important;
    letter-spacing: 0.06em;
    text-shadow: 2px 2px 0 #000, 0 0 12px rgba(212, 175, 55, 0.25);
}
h1 {
    border-bottom: 3px double #8b6f2c;
    padding-bottom: 12px;
    text-align: center;
    font-size: 2.4rem !important;
}

.stApp p, .stApp label, .stApp span:not([data-testid="stMetricValue"] *),
.stApp .stMarkdown, .stApp .stCaption {
    font-family: 'IM Fell English', 'Garamond', serif !important;
    font-size: 1.02rem;
}

[data-testid="stMetricValue"] {
    color: #d4af37 !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #c9a876 !important;
    font-family: 'IM Fell English', serif !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a130b 0%, #0d0905 100%) !important;
    border-right: 2px solid #5a4520;
}

.stButton > button, .stDownloadButton > button {
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-radius: 2px !important;
    transition: all 120ms ease-in-out;
}
button[kind="primary"] {
    background: linear-gradient(180deg, #8b1f1f 0%, #5a1010 100%) !important;
    border: 2px solid #d4af37 !important;
    color: #f4e4b8 !important;
    box-shadow: inset 0 1px 0 rgba(255,220,150,0.2), 0 2px 6px rgba(0,0,0,0.5);
}
button[kind="primary"]:hover {
    background: linear-gradient(180deg, #a82a2a 0%, #6b1818 100%) !important;
    box-shadow: 0 0 14px rgba(212, 175, 55, 0.5);
}
button[kind="secondary"], .stDownloadButton > button {
    background: linear-gradient(180deg, #3a2d1a 0%, #251a0d 100%) !important;
    border: 1px solid #8b6f2c !important;
    color: #e6d4a8 !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #8b6f2c;
    border-radius: 4px;
    background: rgba(40, 30, 18, 0.55);
    padding: 10px;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1a130b;
    border-bottom: 2px solid #5a4520;
}
.stTabs [data-baseweb="tab"] {
    color: #c9a876 !important;
    font-family: 'Cinzel', serif !important;
    letter-spacing: 0.06em;
}
.stTabs [aria-selected="true"] {
    color: #d4af37 !important;
    border-bottom: 3px solid #d4af37 !important;
}

[data-testid="stDataFrame"] {
    border: 2px solid #5a4520;
    border-radius: 2px;
}

.stAlert {
    border-radius: 2px !important;
    border-width: 2px !important;
    font-family: 'IM Fell English', serif !important;
}

input, textarea, .stNumberInput input, .stTextInput input {
    background: #1c150d !important;
    color: #e6d4a8 !important;
    border: 1px solid #8b6f2c !important;
    font-family: 'IM Fell English', serif !important;
}

hr { border-color: #5a4520 !important; }

.flavor {
    font-family: 'IM Fell English', serif;
    font-style: italic;
    color: #b89860;
    text-align: center;
    margin-top: -8px;
    margin-bottom: 18px;
}
</style>
"""
if not boring_mode:
    st.markdown(DAGGERFALL_CSS, unsafe_allow_html=True)

if boring_mode:
    st.title("Intent Send Generator")
    st.caption("Generate the weekly send list from a Bombora intent file.")
else:
    st.title("⚔  The Intent Forge  ⚔")
    st.markdown(
        "<div class='flavor'>Hark, traveler — by Bombora's grace, dispatch the weekly decree.</div>",
        unsafe_allow_html=True,
    )

conn = get_conn()
contact_count = conn.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
sent_count = conn.execute("SELECT COUNT(*) FROM sent").fetchone()[0]


@st.dialog(
    "How to use this tool" if boring_mode else "⚜  A Word from the Steward  ⚜",
    width="large",
)
def show_intro_scroll() -> None:
    if boring_mode:
        st.markdown(
            """
**Quick start:**

---

**1. Upload your contact pool** — In the *Contact Pool* tab, upload your Biscred
contact roster (CSV or XLSX). Choose **Replace** to start fresh, or **Upsert
(merge by uuid)** to merge new contacts into the existing pool.

**2. Generate a send** — In the *Generate Send* tab, upload the week's Bombora
intent file. Set the total number of contacts to send and the cap per firm.

**3. Click Generate** — The app picks contacts that haven't been sent before,
prioritized by Bombora score.

**4. Download the CSV** — Export the resulting send list.

**5. Mark as sent** — Once the send is delivered, mark those contacts as sent
so they won't be picked again in future weeks.

---

The *Sent History* tab records every contact you've sent.
            """
        )
        if st.button("Get started", type="primary", use_container_width=True):
            st.rerun()
    else:
        st.markdown(
            """
**Hark, traveler!** Heed this counsel ere thou dost begin.

---

**1. Inscribe the Roster** — In the *Roster of Souls* tab, bind thy Biscred contact
roster (CSV or XLSX) to the vault. Choose **Replace** to begin anew, or **Upsert
(merge by uuid)** to fold new names into the existing throng.

**2. Forge a Decree** — In the *Forge a Decree* tab, present the week's Bombora
codex of intent. Decree how many souls shall be dispatched, and the cap per
banner-house, that no liege be overburdened.

**3. Sound the Horn** — Strike the horn, and the forge shall divine which souls
to summon. Names already sealed in the tome shall be spared.

**4. Take the Scroll** — Bear away the resulting parchment of contacts.

**5. Affix the Royal Seal** — Once thy decrees are delivered, swear the deed done
and seal those souls in the tome, lest they be summoned again in weeks to come.

---

The *Tome of the Sealed* records every soul thou hast ever called. The hearth
crackles in the sidebar — silence it shouldst silence be thy will.
            """
        )
        if st.button("Begin thy quest", type="primary", use_container_width=True):
            st.rerun()


if "intro_seen" not in st.session_state:
    st.session_state["intro_seen"] = True
    show_intro_scroll()

with st.sidebar:
    st.toggle("Boring Mode", key="boring_mode")
    st.caption(
        t(
            "Banish the medieval fancy and don a more traditional vestment.",
            "Switch to the medieval theme.",
        )
    )
    st.divider()
    st.header(t("⚜  Ledger", "Status"))
    st.metric(t("Souls in the Roster", "Contacts in pool"), f"{contact_count:,}")
    st.metric(t("Decrees Sealed", "Contacts sent"), f"{sent_count:,}")
    st.caption(f"{t('Vault', 'Database')}: `{DB_PATH}`")
    st.divider()
    if boring_mode:
        hearth_lit = False
    else:
        hearth_lit = st.toggle("🔥  Kindle the Hearth", value=True, key="hearth")
        st.caption(
            "The hearth crackles whilst thou laborest. "
            "Snuff it shouldst silence be thy will."
        )
    if st.button(
        t("📜  Steward's Counsel", "Help"),
        use_container_width=True,
    ):
        show_intro_scroll()

render_hearth(hearth_lit)

tab_gen, tab_pool, tab_sent = st.tabs(
    [
        t("⚔  Forge a Decree", "Generate Send"),
        t("📜  Roster of Souls", "Contact Pool"),
        t("🕯  Tome of the Sealed", "Sent History"),
    ]
)

# ---- Tab: Contact pool ----
with tab_pool:
    st.subheader(t("Inscribe the Roster of Souls", "Upload Contact Pool"))
    st.caption(
        t(
            "The Biscred contact roster is bound to the local vault. "
            "Replace the parchment outright, or merge new names by uuid.",
            "Upload your Biscred contact roster to the local database. "
            "Replace the existing pool, or merge new contacts by uuid.",
        )
    )
    pool_file = st.file_uploader(
        t("Biscred roster (CSV or XLSX)", "Contact pool file (CSV or XLSX)"),
        type=["csv", "xlsx", "xls"],
        key="pool",
    )
    if not boring_mode:
        _play_scroll_if_new(pool_file, "pool")
    mode = st.radio("Mode", ["Replace", "Upsert (merge by uuid)"], horizontal=True)

    if pool_file and st.button(
        t("Bind to the Vault", "Upload"), type="primary"
    ):
        try:
            df = read_table(pool_file)
            n = load_contact_pool(conn, df, replace=(mode == "Replace"))
            st.success(
                t(
                    f"⚜ {n:,} souls bound to the vault.",
                    f"Loaded {n:,} contacts.",
                )
            )
            st.rerun()
        except Exception as e:
            st.error(
                t(f"The binding failed: {e}", f"Upload failed: {e}")
            )

# ---- Tab: Generate ----
with tab_gen:
    st.subheader(t("Forge this Week's Decree", "Generate this Week's Send"))

    if contact_count == 0:
        st.warning(
            t(
                "By Mara's mercy — no roster yet inscribed! "
                "Visit the **Roster of Souls** to bind one.",
                "No contact pool loaded yet. "
                "Go to the **Contact Pool** tab to upload one.",
            )
        )
    else:
        bombora_file = st.file_uploader(
            t(
                "Bombora codex of intent (CSV or XLSX)",
                "Bombora intent file (CSV or XLSX)",
            ),
            type=["csv", "xlsx", "xls"],
            key="bombora",
        )
        if not boring_mode:
            _play_scroll_if_new(bombora_file, "bombora")

        c1, c2, c3 = st.columns(3)
        total_target = c1.number_input(
            t("Souls to dispatch", "Total contacts to send"),
            min_value=1, value=200, step=10,
        )
        max_per_firm = c2.number_input(
            t("Max per banner-house", "Max per firm"),
            min_value=1, value=12, step=1,
        )
        seed_input = c3.text_input(
            t("Augury seed (optional)", "Random seed (optional)"),
            value="",
        )

        if bombora_file and st.button(
            t("Sound the Horn", "Generate"), type="primary"
        ):
            if not boring_mode:
                sound_the_horn()
            try:
                bom = read_table(bombora_file)
                seed = int(seed_input) if seed_input.strip() else None
                result, stats = generate_send(
                    conn, bom, total_target=int(total_target),
                    max_per_firm=int(max_per_firm), seed=seed,
                )
                st.session_state["generated"] = result
                st.session_state["generated_stats"] = stats
                st.session_state["generated_at"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                st.error(f"Failed to generate: {e}")
                st.session_state.pop("generated", None)

        if "generated" in st.session_state:
            result: pd.DataFrame = st.session_state["generated"]
            stats: dict = st.session_state["generated_stats"]
            ts: str = st.session_state["generated_at"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(t("Souls", "Contacts"), f"{stats['rows']:,}")
            m2.metric(
                t("Banner-houses", "Firms"),
                f"{result['company_name'].nunique():,}",
            )
            m3.metric(
                t("Houses matched", "Firms matched"),
                f"{stats['firms_matched']:,}",
            )
            m4.metric(
                t("Houses unmanned", "Firms skipped"),
                f"{stats['firms_skipped']:,}",
            )

            if stats["skipped_examples"]:
                with st.expander(
                    t(
                        "Houses with no souls available",
                        "Firms with no available contacts",
                    )
                ):
                    st.write(stats["skipped_examples"])

            st.dataframe(result, use_container_width=True, height=400)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                t("Take the Scroll", "Download CSV"),
                csv,
                file_name=f"send_{ts}.csv",
                mime="text/csv",
            )

            st.divider()
            st.caption(
                t(
                    "Once the decree is delivered, affix the royal seal so these souls "
                    "are not summoned again in future weeks.",
                    "Once the send is delivered, mark these contacts as sent so they "
                    "won't be picked again in future weeks.",
                )
            )
            confirm = st.checkbox(
                t("Aye, the deed is done", "Confirm send delivered"),
                key="confirm_sent",
            )
            if confirm and st.button(
                t("Affix the Royal Seal", "Mark as sent"), type="secondary"
            ):
                now = datetime.utcnow().isoformat()
                conn.executemany(
                    "INSERT OR REPLACE INTO sent (uuid, sent_at) VALUES (?, ?)",
                    [(u, now) for u in result["uuid"].tolist()],
                )
                conn.commit()
                st.success(
                    t(
                        f"⚜ {len(result):,} souls sealed in the tome.",
                        f"{len(result):,} contacts marked as sent.",
                    )
                )
                for k in ("generated", "generated_stats", "generated_at", "confirm_sent"):
                    st.session_state.pop(k, None)
                st.rerun()

# ---- Tab: Sent history ----
with tab_sent:
    st.subheader(t("Tome of the Sealed", "Sent History"))
    df = pd.read_sql(
        "SELECT s.sent_at, s.uuid, c.contact_name, c.contact_email, c.company_name "
        "FROM sent s LEFT JOIN contacts c ON s.uuid = c.uuid "
        "ORDER BY s.sent_at DESC",
        conn,
    )
    st.write(
        t(
            f"Souls inscribed in the tome: **{len(df):,}**",
            f"Total contacts sent: **{len(df):,}**",
        )
    )
    st.dataframe(df, use_container_width=True, height=500)

    if len(df) > 0:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("Transcribe the Tome", "Download history CSV"),
            csv,
            file_name="sent_history.csv",
            mime="text/csv",
        )

        with st.expander(t("⚠ The Forbidden Rite", "⚠ Danger zone")):
            st.caption(
                t(
                    "Burn the tome and forget all who came before. The roster remains.",
                    "Clear all sent history. The contact pool is unaffected.",
                )
            )
            confirm_clear = st.text_input(
                t(
                    'Speak the word "CLEAR" to invoke the rite',
                    'Type "CLEAR" to confirm',
                )
            )
            if confirm_clear == "CLEAR" and st.button(
                t("Burn the Tome", "Clear sent history")
            ):
                conn.execute("DELETE FROM sent")
                conn.commit()
                st.success(
                    t(
                        "The tome is ash. The vault forgets.",
                        "Sent history cleared.",
                    )
                )
                st.rerun()
