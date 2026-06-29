import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PersonaIQ · Personality Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #07070d;
    color: #dde1f0;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 2rem 6rem !important;
    max-width: 1120px !important;
}

/* ── Hero ── */
.hero-wrap {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    padding: 3.5rem 3rem 3rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(145deg, #0d0d1a 0%, #0a0a16 100%);
    border: 1px solid #1c1c36;
}
.hero-glow {
    position: absolute;
    top: -80px; right: -80px;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(108,99,255,0.18) 0%, transparent 65%);
    pointer-events: none;
}
.hero-glow-2 {
    position: absolute;
    bottom: -60px; left: 100px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(168,85,247,0.1) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.3);
    color: #9d94ff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.4rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #f0f0ff 20%, #9d94ff 60%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #7a82a0;
    line-height: 1.75;
    max-width: 500px;
    margin-bottom: 2rem;
}
.hero-stats {
    display: flex;
    gap: 0;
    border: 1px solid #1c1c36;
    border-radius: 14px;
    overflow: hidden;
    width: fit-content;
    background: rgba(255,255,255,0.02);
}
.stat {
    padding: 0.9rem 2rem;
    border-right: 1px solid #1c1c36;
    text-align: center;
}
.stat:last-child { border-right: none; }
.stat-n {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #b8b0ff;
    line-height: 1;
}
.stat-l {
    font-size: 0.68rem;
    color: #3e4460;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Section headers ── */
.sec-hdr {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #111122;
}
.sec-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(108,99,255,0.5);
}
.sec-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #6c63ff;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.sec-emoji {
    font-size: 1rem;
    margin-left: auto;
    opacity: 0.4;
}

/* ── Slider card wrapper ── */
.slider-card {
    background: #0d0d1a;
    border: 1px solid #151530;
    border-radius: 14px;
    padding: 1.1rem 1.2rem 0.6rem;
    margin-bottom: 0;
    transition: border-color 0.2s;
}
.slider-card:hover {
    border-color: #252545;
}

/* ── Streamlit slider overrides ── */
div[data-testid="stSlider"] {
    padding: 0.05rem 0 0.5rem !important;
}
div[data-testid="stSlider"] > label {
    font-size: 0.76rem !important;
    color: #5a6080 !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 0.4rem !important;
    display: block !important;
}
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-size: 0.62rem !important;
    color: #2e3450 !important;
    font-family: 'DM Mono', monospace !important;
}
/* Slider track */
div[data-testid="stSlider"] > div > div > div {
    background: #1a1a2e !important;
}
/* Slider thumb */
div[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(135deg, #6c63ff, #9c3fe8) !important;
    box-shadow: 0 0 12px rgba(108,99,255,0.5) !important;
    border: 2px solid rgba(255,255,255,0.1) !important;
}

/* ── Column gaps ── */
div[data-testid="column"] {
    padding: 0 0.5rem !important;
}
div[data-testid="column"]:first-child { padding-left: 0 !important; }
div[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* ── Submit button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #6c63ff 0%, #9c3fe8 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 8px 40px rgba(108,99,255,0.4) !important;
    transition: all 0.2s ease !important;
    margin-top: 1.5rem !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 12px 55px rgba(108,99,255,0.6) !important;
    transform: translateY(-2px) !important;
}
div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result card ── */
.result-outer {
    margin-top: 2.5rem;
    border-radius: 24px;
    border: 1px solid #1c1c36;
    background: #0d0d1a;
    overflow: hidden;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-header {
    padding: 2rem 2.5rem 1.75rem;
    border-bottom: 1px solid #111122;
    position: relative;
    overflow: hidden;
}
.result-glow {
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    pointer-events: none;
    opacity: 0.9;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3e4460;
    margin-bottom: 0.7rem;
}
.result-type {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4.5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
}
.result-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}
.result-col {
    padding: 2rem 2.5rem;
}
.result-col:first-child {
    border-right: 1px solid #111122;
}
.col-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3e4460;
    margin-bottom: 0.9rem;
    margin-top: 0;
}
.confidence-num {
    font-family: 'DM Mono', monospace;
    font-size: 3.2rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 0.7rem;
}
.conf-bar-bg {
    background: #0f0f1e;
    border-radius: 100px;
    height: 6px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s ease;
}
.conf-note {
    font-size: 0.71rem;
    color: #2e3450;
    margin-bottom: 0.3rem;
}
.traits-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 1.6rem;
}
.trait-chip {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.28rem 0.8rem;
    border-radius: 100px;
    letter-spacing: 0.02em;
}
.desc-text {
    font-size: 0.88rem;
    color: #7a82a0;
    line-height: 1.8;
    margin-bottom: 1.8rem;
}
.prob-item {
    display: grid;
    grid-template-columns: 110px 1fr 52px;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #0c0c1c;
}
.prob-item:last-child { border-bottom: none; }
.prob-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #dde1f0;
}
.prob-bar-bg {
    background: #0f0f1e;
    border-radius: 100px;
    height: 5px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
}
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    text-align: right;
}

.prob-wrap {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ── Color themes ── */
.clr-introvert { color: #a78bfa; }
.clr-ambivert  { color: #34d399; }
.clr-extrovert { color: #fbbf24; }

.bar-introvert { background: linear-gradient(90deg, #6c63ff, #a78bfa); }
.bar-ambivert  { background: linear-gradient(90deg, #059669, #34d399); }
.bar-extrovert { background: linear-gradient(90deg, #d97706, #fbbf24); }

.chip-introvert { background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.25); color: #a78bfa; }
.chip-ambivert  { background: rgba(52,211,153,0.1);  border: 1px solid rgba(52,211,153,0.25);  color: #34d399; }
.chip-extrovert { background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.25);  color: #fbbf24; }

.glow-introvert { background: radial-gradient(circle, rgba(108,99,255,0.28) 0%, transparent 70%); }
.glow-ambivert  { background: radial-gradient(circle, rgba(52,211,153,0.22) 0%, transparent 70%); }
.glow-extrovert { background: radial-gradient(circle, rgba(251,191,36,0.22) 0%, transparent 70%); }

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid #1c1c36 !important;
    border-radius: 14px !important;
    background: #0a0a14 !important;
    margin-top: 1.2rem !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    color: #4a5070 !important;
    font-size: 0.82rem !important;
    padding: 0.9rem 1.2rem !important;
}
div[data-testid="stExpander"] summary:hover { color: #7a82a0 !important; }

/* ── Warning / info boxes ── */
.warn-box {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #b89a40;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.warn-box strong { color: #fbbf24; }

/* ── Debug panel ── */
.debug-inner {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #5a6080;
    line-height: 1.8;
}
.debug-inner b { color: #9d94ff; }

/* ── Responsive ── */
@media (max-width: 760px) {
    .block-container { padding: 1.5rem 1rem 5rem !important; }
    .hero-wrap { padding: 2.5rem 1.5rem 2rem; }
    .result-body { grid-template-columns: 1fr; }
    .result-col:first-child { border-right: none; border-bottom: 1px solid #111122; }
    .result-col { padding: 1.5rem 1.5rem; }
    .hero-stats { flex-direction: column; width: 100%; }
    .stat { border-right: none !important; border-bottom: 1px solid #1c1c36; }
    .stat:last-child { border-bottom: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("personality_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ── Feature config ─────────────────────────────────────────────────────────────
# Must match training scaler feature order exactly.
FEATURE_ORDER = [
    "social_energy",
    "alone_time_preference",
    "talkativeness",
    "deep_reflection",
    "group_comfort",
    "party_liking",
    "listening_skill",
    "empathy",
    "organization",
    "leadership",
    "risk_taking",
    "public_speaking_comfort",
    "curiosity",
    "routine_preference",
    "excitement_seeking",
    "friendliness",
    "planning",
    "spontaneity",
    "adventurousness",
    "reading_habit",
    "sports_interest",
    "online_social_usage",
    "travel_desire",
    "gadget_usage",
    "work_style_collaborative",
    "decision_speed",
]


PERSONA = {
    "Introvert": {
        "emoji": "🔮",
        "color_cls": "introvert",
        "description": (
            "You draw energy from within. A deep thinker and careful listener, you thrive "
            "in focused, meaningful environments. You prefer quality over quantity in "
            "relationships and process the world internally before sharing your perspective."
        ),
        "traits": ["Deep Thinker", "Self-Aware", "Great Listener", "Independent", "Thoughtful"],
    },
    "Ambivert": {
        "emoji": "⚡",
        "color_cls": "ambivert",
        "description": (
            "You're the bridge between worlds. Effortlessly adaptive — energised by social "
            "connection when the moment calls for it, equally comfortable recharging in "
            "solitude. You read rooms intuitively and know exactly when to lead or step back."
        ),
        "traits": ["Adaptable", "Balanced", "Empathetic", "Versatile", "Perceptive"],
    },
    "Extrovert": {
        "emoji": "🌟",
        "color_cls": "extrovert",
        "description": (
            "You energise every room you enter. You thrive on connection, collaboration, "
            "and new experiences. You think out loud, build wide networks naturally, and "
            "bring enthusiasm that lifts everyone around you. Social situations are where "
            "you do your best work."
        ),
        "traits": ["Energetic", "Charismatic", "Team Player", "Expressive", "Action-Oriented"],
    },
}


def build_label_map(model):
    """Return (label_map, map_type).

    label_map maps numeric class values (e.g., 0/1/2) to one of:
    {"Introvert","Ambivert","Extrovert"}
    """

    classes = model.classes_

    # If model already uses string labels
    if all(isinstance(c, str) for c in classes):
        mapping = {}
        for c in classes:
            cl = str(c).strip()
            if cl.lower() in ("extravert", "extraversion", "extraverted", "e"):
                mapping[cl] = "Extrovert"
            elif cl.lower() in ("introvert", "introversion", "i"):
                mapping[cl] = "Introvert"
            else:
                mapping[cl] = "Ambivert"
        # For our callers we only need numeric map; keep output compatible.
        return {i: v for i, v in enumerate(["Introvert", "Ambivert", "Extrovert"])}, "string-model"

    # Numeric labels: infer which one corresponds to extremes.
    try:
        n_features = len(FEATURE_ORDER)
        all_high_df = pd.DataFrame([[10.0] * n_features], columns=FEATURE_ORDER)
        all_low_df = pd.DataFrame([[1.0] * n_features], columns=FEATURE_ORDER)

        proba_high = model.predict_proba(scaler.transform(all_high_df))[0]
        proba_low = model.predict_proba(scaler.transform(all_low_df))[0]

        ext_idx = int(np.argmax(proba_high))
        intro_idx = int(np.argmax(proba_low))

        remaining = [i for i in range(len(classes)) if i not in (ext_idx, intro_idx)]
        amb_idx = remaining[0] if remaining else int(np.setdiff1d(range(len(classes)), [ext_idx, intro_idx])[0])

        ext_cls_val = int(classes[ext_idx])
        intro_cls_val = int(classes[intro_idx])
        amb_cls_val = int(classes[amb_idx])

        return {intro_cls_val: "Introvert", amb_cls_val: "Ambivert", ext_cls_val: "Extrovert"}, "int-inferred"
    except Exception:
        pass

    # Common sklearn assumption
    try:
        int_classes = [int(c) for c in classes]
        if sorted(int_classes) == [0, 1, 2]:
            return {0: "Introvert", 1: "Ambivert", 2: "Extrovert"}, "int-default"
    except Exception:
        pass

    names = ["Introvert", "Ambivert", "Extrovert"]
    return {int(c): names[i % 3] for i, c in enumerate(classes)}, "int-fallback"


def slider_field(label: str, key: str, default: int = 5) -> int:
    return st.slider(label, min_value=1, max_value=10, value=default, key=key)


def section_header(title: str, emoji: str):
    st.markdown(
        f"""
        <div class="sec-hdr">
            <div class="sec-dot"></div>
            <span class="sec-name">{title}</span>
            <span class="sec-emoji">{emoji}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-glow-2"></div>
    <div class="hero-eyebrow">🧬 &nbsp;ML-Powered Personality Analysis</div>
    <h1 class="hero-title">Discover Your<br>True Personality</h1>
    <p class="hero-sub">
        Rate yourself honestly across 26 behavioural dimensions. Our model analyses
        your responses and classifies your personality type in under a second.
    </p>
    <div class="hero-stats">
        <div class="stat">
            <div class="stat-n">26</div>
            <div class="stat-l">Dimensions</div>
        </div>
        <div class="stat">
            <div class="stat-n">3</div>
            <div class="stat-l">Types</div>
        </div>
        <div class="stat">
            <div class="stat-n">99.75%</div>
            <div class="stat-l">Accuracy</div>
        </div>
        <div class="stat">
            <div class="stat-n"><1s</div>
            <div class="stat-l">Inference</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Model error guard ─────────────────────────────────────────────────────────
if not model_loaded:
    st.markdown(
        f"""
    <div class="warn-box">
        <strong>⚠️  Model files not found.</strong><br>
        Make sure <code>personality_model.pkl</code> and <code>scaler.pkl</code> are in the
        same directory as this script.<br><br>
        Error: <code>{model_error}</code>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("persona_form", border=False):

    section_header("Social Traits", "👥")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        social_energy = slider_field("Social Energy", "social_energy")
    with c2:
        talkativeness = slider_field("Talkativeness", "talkativeness")
    with c3:
        group_comfort = slider_field("Group Comfort", "group_comfort")
    with c4:
        friendliness = slider_field("Friendliness", "friendliness")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        party_liking = slider_field("Party Liking", "party_liking")
    with c2:
        public_speaking_comfort = slider_field("Public Speaking", "public_speaking_comfort")
    with c3:
        online_social_usage = slider_field("Online Social Usage", "online_social_usage")
    with c4:
        work_style_collaborative = slider_field("Collaborative Style", "work_style_collaborative")

    section_header("Inner World & Reflection", "🧠")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alone_time_preference = slider_field("Alone Time", "alone_time_preference")
    with c2:
        deep_reflection = slider_field("Deep Reflection", "deep_reflection")
    with c3:
        curiosity = slider_field("Curiosity", "curiosity")
    with c4:
        reading_habit = slider_field("Reading Habit", "reading_habit")

    section_header("Interpersonal Skills", "💬")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        empathy = slider_field("Empathy", "empathy")
    with c2:
        listening_skill = slider_field("Listening Skill", "listening_skill")
    with c3:
        leadership = slider_field("Leadership", "leadership")
    with c4:
        decision_speed = slider_field("Decision Speed", "decision_speed")

    section_header("Work & Structure", "💼")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        organization = slider_field("Organization", "organization")
    with c2:
        planning = slider_field("Planning", "planning")
    with c3:
        routine_preference = slider_field("Routine Preference", "routine_preference")
    with c4:
        spontaneity = slider_field("Spontaneity", "spontaneity")

    section_header("Adventure & Risk", "🚀")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        risk_taking = slider_field("Risk Taking", "risk_taking")
    with c2:
        excitement_seeking = slider_field("Excitement Seeking", "excitement_seeking")
    with c3:
        adventurousness = slider_field("Adventurousness", "adventurousness")
    with c4:
        travel_desire = slider_field("Travel Desire", "travel_desire")

    section_header("Lifestyle & Interests", "🎯")
    c1, c2, _, _ = st.columns(4)
    with c1:
        sports_interest = slider_field("Sports Interest", "sports_interest")
    with c2:
        gadget_usage = slider_field("Gadget Usage", "gadget_usage")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("⚡  Analyse My Personality", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:

    values = {
        "social_energy": social_energy,
        "alone_time_preference": alone_time_preference,
        "talkativeness": talkativeness,
        "deep_reflection": deep_reflection,
        "group_comfort": group_comfort,
        "party_liking": party_liking,
        "listening_skill": listening_skill,
        "empathy": empathy,
        "organization": organization,
        "leadership": leadership,
        "risk_taking": risk_taking,
        "public_speaking_comfort": public_speaking_comfort,
        "curiosity": curiosity,
        "routine_preference": routine_preference,
        "excitement_seeking": excitement_seeking,
        "friendliness": friendliness,
        "planning": planning,
        "spontaneity": spontaneity,
        "adventurousness": adventurousness,
        "reading_habit": reading_habit,
        "sports_interest": sports_interest,
        "online_social_usage": online_social_usage,
        "travel_desire": travel_desire,
        "gadget_usage": gadget_usage,
        "work_style_collaborative": work_style_collaborative,
        "decision_speed": decision_speed,
    }

    input_df = pd.DataFrame([[values[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(input_df)

    proba = model.predict_proba(scaled)[0]
    pred_raw = model.predict(scaled)[0]

    label_map, map_type = build_label_map(model)

    # Resolve predicted label
    if isinstance(pred_raw, str):
        pred_str = pred_raw.strip().capitalize()
        if pred_str in ("Extravert", "Extraversion", "E"):
            pred_str = "Extrovert"
        if pred_str in ("Introversion", "I"):
            pred_str = "Introvert"
        if pred_str in ("A", "Both", "Neither"):
            pred_str = "Ambivert"
        prediction = pred_str if pred_str in PERSONA else "Ambivert"
    else:
        prediction = label_map.get(int(pred_raw), "Ambivert")

    confidence = round(float(max(proba)) * 100, 1)
    cfg = PERSONA.get(prediction, PERSONA["Ambivert"])
    cc = cfg["color_cls"]

    # Probability rows
    classes_named = []
    for raw_cls, p in zip(model.classes_, proba):
        if isinstance(raw_cls, str):
            lbl = raw_cls.strip().capitalize()
            if lbl in ("Extravert", "Extraversion", "E"):
                lbl = "Extrovert"
            if lbl in ("Introversion", "I"):
                lbl = "Introvert"
            if lbl in ("A", "Both", "Neither"):
                lbl = "Ambivert"
        else:
            lbl = label_map.get(int(raw_cls), str(raw_cls))
        classes_named.append((lbl, round(float(p) * 100, 1)))

    classes_named.sort(key=lambda x: -x[1])

    trait_chips = "".join(
        f'<span class="trait-chip chip-{cc}">{t}</span>' for t in cfg["traits"]
    )

    prob_rows = ""
    for name, pct in classes_named:
        nc = PERSONA.get(name, {}).get("color_cls", cc)
        emoji = PERSONA.get(name, {}).get("emoji", "•")
        prob_rows += (
            f'<div class="prob-item">'
            f'<span class="prob-name">{emoji} {name}</span>'
            f'<div class="prob-bar-bg">'
            f'  <div class="prob-bar-fill bar-{nc}" style="width:{pct}%"></div>'
            f'</div>'
            f'<span class="prob-pct clr-{nc}">{pct}%</span>'
            f'</div>'
        )

    st.markdown(
        f"""
    <div class="result-outer">
        <div class="result-header">
            <div class="result-glow glow-{cc}"></div>
            <div class="result-label">Your personality type</div>
            <div class="result-type clr-{cc}">{cfg['emoji']} {prediction}</div>
        </div>
        <div class="result-body">
            <div class="result-col">
                <div class="col-label">Confidence Score</div>
                <div class="confidence-num clr-{cc}">{confidence}%</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill bar-{cc}" style="width:{confidence}%"></div>
                </div>
                <div class="conf-note">Based on your 26-dimension profile</div>
                <div class="traits-wrap">{trait_chips}</div>
            </div>
            <div class="result-col">
                <div class="col-label">What this means</div>
                <p class="desc-text">{cfg['description']}</p>
                <div class="col-label">Probability breakdown</div>
                <div class="prob-wrap">{prob_rows}</div>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.expander("📋  View your full input summary"):
        display_df = pd.DataFrame(
            {"Feature": list(values.keys()), "Your Score": list(values.values())}
        ).set_index("Feature")
        # Avoid pandas Styler background gradients on Streamlit Cloud
        # because it requires matplotlib (not always installed there).
        st.dataframe(display_df, use_container_width=True)

    with st.expander("🔧  Debug — Model internals (check this if result seems wrong)"):
        raw_classes_str = ", ".join(f"{i}→'{c}'" for i, c in enumerate(model.classes_))
        st.markdown(
            f"""
        <div class="debug-inner">
            <b>model.classes_</b> : [{raw_classes_str}]<br>
            <b>Label map used</b> : {label_map}  <i>({map_type})</i><br>
            <b>Raw prediction</b> : {pred_raw}  (type: {type(pred_raw).__name__})<br>
            <b>Resolved label</b> : {prediction}<br>
            <b>Probabilities</b>  : {{ {', '.join([f"{n}: {p}%" for n, p in classes_named])} }}<br>
            <b>Feature order</b>  : {FEATURE_ORDER}<br><br>
            <i>If the result is wrong, paste model.classes_ output here
            and share with the developer to fix LABEL_MAP.</i>
        </div>
        """,
            unsafe_allow_html=True,
        )

