import re
import pandas as pd
import pdfplumber
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Syllabus to Study Plan",
    page_icon="📘",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

:root {
    --sky-top: #87CEEB;
    --sky-mid: #B9E3F7;
    --sky-bottom: #E6F5FD;

    --ink: #0B3954;
    --ink-soft: #35607A;

    --brand: #0B5CAD;
    --brand-dark: #084A8C;

    --panel: rgba(255, 255, 255, 0.94);
    --line: rgba(11, 57, 84, 0.14);
}

/* -----------------------------------------------------------------------
   Page Background
----------------------------------------------------------------------- */

[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: linear-gradient(
        180deg,
        var(--sky-top) 0%,
        var(--sky-mid) 38%,
        var(--sky-bottom) 100%
    ) fixed;
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}

.block-container {
    max-width: 880px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* -----------------------------------------------------------------------
   General Font
----------------------------------------------------------------------- */

html,
body,
[data-testid="stApp"],
.stMarkdown,
label,
p,
span,
div {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
    color: var(--ink);
}

/* -----------------------------------------------------------------------
   Main Heading
----------------------------------------------------------------------- */

.app-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-weight: 800;
    font-size: 3.2rem;
    line-height: 1.15;
    text-align: center;
    margin: 0 0 0.5rem 0;

    background: linear-gradient(
        90deg,
        #084A8C,
        #0B5CAD,
        #087E8B
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* -----------------------------------------------------------------------
   Subtitle
----------------------------------------------------------------------- */

.app-sub {
    font-family: 'Poppins', sans-serif;
    font-size: 1.05rem;
    text-align: center;
    color: var(--ink-soft);
    max-width: 650px;
    margin: 0 auto 2rem auto;
    line-height: 1.7;
}

/* -----------------------------------------------------------------------
   Section Heading
----------------------------------------------------------------------- */

.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-weight: 700;
    font-size: 2rem;
    color: var(--brand-dark);
    margin: 2rem 0 1rem 0;
}

/* -----------------------------------------------------------------------
   Input Panel
----------------------------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--line) !important;
    border-radius: 20px;
    padding: 1rem 1.2rem;
    box-shadow: 0 8px 28px rgba(11, 57, 84, 0.14);
}

/* -----------------------------------------------------------------------
   File Uploader
----------------------------------------------------------------------- */

[data-testid="stFileUploaderDropzone"] {
    background: #F2FAFE;
    border: 2px dashed #5BAFD8;
    border-radius: 14px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--brand);
}

/* -----------------------------------------------------------------------
   Text Area / Number Input
----------------------------------------------------------------------- */

.stTextArea textarea,
.stNumberInput input {
    background: #FFFFFF;
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 12px;
}

/* -----------------------------------------------------------------------
   Buttons
----------------------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {
    background: var(--brand);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.6rem;
    font-weight: 700;
    font-size: 1rem;
    transition: 0.2s ease;
}

.stButton > button p,
.stDownloadButton > button p {
    color: #FFFFFF;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--brand-dark);
    transform: translateY(-2px);
}

/* -----------------------------------------------------------------------
   Metrics
----------------------------------------------------------------------- */

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 5px 18px rgba(11, 57, 84, 0.10);
}

[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', Georgia, serif;
    color: var(--brand);
}

/* -----------------------------------------------------------------------
   Table
----------------------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    background: #FFFFFF;
}

/* -----------------------------------------------------------------------
   Mobile
----------------------------------------------------------------------- */

@media (max-width: 640px) {

    .app-title {
        font-size: 2.3rem;
    }

    .app-sub {
        font-size: 0.95rem;
    }

    .block-container {
        padding-top: 2rem;
    }
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Study-plan logic
# ---------------------------------------------------------------------------

HARD = [
    "algorithm",
    "analysis",
    "design",
    "optimization",
    "architecture",
    "proof",
    "complexity",
    "model",
    "network",
    "protocol",
    "neural",
    "probability",
    "transform",
    "advanced",
    "synthesis"
]

EASY = [
    "introduction",
    "overview",
    "basics",
    "basic",
    "fundamentals",
    "history",
    "definition",
    "concepts",
    "applications",
    "types"
]

WEIGHT = {
    "Easy": 1,
    "Medium": 2,
    "Hard": 3
}


# ---------------------------------------------------------------------------
# Extract text from PDF
# ---------------------------------------------------------------------------

def extract_text(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    return text


# ---------------------------------------------------------------------------
# Split syllabus into units
# ---------------------------------------------------------------------------

def split_units(text):

    """
    Split syllabus on headings like:
    Unit I
    UNIT-2
    Unit 3:
    """

    pattern = r"(?im)^\s*unit[\s\-:]*([ivx\d]+)\b[^\n]*"

    matches = list(re.finditer(pattern, text))

    if not matches:
        return {"Syllabus": text}

    units = {}

    for i, m in enumerate(matches):

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        name = "Unit " + m.group(1).upper()

        units[name] = text[m.end():end]

    return units


# ---------------------------------------------------------------------------
# Split topics
# ---------------------------------------------------------------------------

def split_topics(block):

    block = re.sub(r"\s+", " ", block)

    parts = re.split(
        r"[;,•\u2022\n]| - ",
        block
    )

    topics = [
        p.strip(" .:-")
        for p in parts
    ]

    return [
        t for t in topics
        if 3 <= len(t) <= 80
    ]


# ---------------------------------------------------------------------------
# Determine difficulty
# ---------------------------------------------------------------------------

def difficulty(topic):

    t = topic.lower()

    if any(k in t for k in HARD) or len(t.split()) > 8:
        return "Hard"

    if any(k in t for k in EASY):
        return "Easy"

    return "Medium"


# ---------------------------------------------------------------------------
# Build study plan
# ---------------------------------------------------------------------------

def build_plan(text, weeks):

    rows = []

    for unit, block in split_units(text).items():

        for topic in split_topics(block):

            rows.append({
                "Unit": unit,
                "Topic": topic,
                "Difficulty": difficulty(topic)
            })

    if not rows:
        return pd.DataFrame()

    total = sum(
        WEIGHT[r["Difficulty"]]
        for r in rows
    )

    target = total / weeks

    week = 1
    load = 0

    for r in rows:

        r["Week"] = week

        load += WEIGHT[r["Difficulty"]]

        if load >= target and week < weeks:

            week += 1
            load = 0

    return pd.DataFrame(rows)[
        [
            "Week",
            "Unit",
            "Topic",
            "Difficulty"
        ]
    ]


# ---------------------------------------------------------------------------
# Table colouring
# ---------------------------------------------------------------------------

DIFFICULTY_STYLE = {

    "Easy":
        "background-color:#D4F1DF;"
        "color:#14633A;"
        "font-weight:700;",

    "Medium":
        "background-color:#FFEBB8;"
        "color:#7A5200;"
        "font-weight:700;",

    "Hard":
        "background-color:#FBD5D5;"
        "color:#8C1D1D;"
        "font-weight:700;"
}


def style_plan(plan):

    styler = plan.style

    paint = (
        getattr(styler, "map", None)
        or styler.applymap
    )

    return paint(
        lambda v: DIFFICULTY_STYLE.get(v, ""),
        subset=["Difficulty"]
    )


# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 class="app-title">📘✨ Syllabus to Study Plan</h1>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="app-sub">
        🎯 Turn your syllabus into a smart, organized and achievable
        <b>week-by-week study plan</b>.
        <br>
        📚 Upload your syllabus, choose your exam timeline,
        and let the planner organize your topics by difficulty.
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

with st.container(border=True):

    st.markdown("### 📄 Upload Your Syllabus")

    upload = st.file_uploader(
        "Choose your syllabus PDF",
        type="pdf"
    )

    text = extract_text(upload) if upload else ""

    st.markdown("### 📝 Or Enter Syllabus Manually")

    text = st.text_area(
        "Paste your syllabus text here",
        value=text,
        height=250,
        placeholder=(
            "Example: Unit 1 - Introduction to Machine Learning..."
        )
    )

    st.markdown("### 📅 Exam Preparation Time")

    weeks = st.number_input(
        "How many weeks do you have until your exam?",
        min_value=1,
        max_value=52,
        value=6
    )

    st.write("")

    generate = st.button(
        "🚀 Generate My Study Plan"
    )


# ---------------------------------------------------------------------------
# Generate Result
# ---------------------------------------------------------------------------

if generate:

    plan = build_plan(
        text,
        int(weeks)
    )

    if plan.empty:

        st.warning(
            "⚠️ No topics found. "
            "Please check your syllabus text or PDF."
        )

    else:

        st.markdown(
            '<h2 class="section-title">'
            '📚✨ Your Personalized Study Plan'
            '</h2>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------------------------
        # Summary metrics
        # -------------------------------------------------------------------

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "📚 Topics",
            len(plan)
        )

        c2.metric(
            "📅 Weeks",
            plan["Week"].nunique()
        )

        c3.metric(
            "🔥 Hard Topics",
            int(
                (plan["Difficulty"] == "Hard").sum()
            )
        )

        st.write("")

        # -------------------------------------------------------------------
        # Study plan table
        # -------------------------------------------------------------------

        st.dataframe(
            style_plan(plan),
            width="stretch",
            hide_index=True
        )

        st.write("")

        # -------------------------------------------------------------------
        # Download button
        # -------------------------------------------------------------------

        st.download_button(
            "📥 Download Study Plan",
            plan.to_csv(index=False),
            "study_plan.csv",
            "text/csv"
        )