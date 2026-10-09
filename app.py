
import streamlit as st
import joblib
import pandas as pd
import re
from scipy.sparse import hstack

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VeriJob AI | Job Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- PREMIUM UI DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d19;
    --panel: #111a2c;
    --border: #25334b;
    --text: #edf3ff;
    --muted: #91a1bb;
    --accent: #6c8cff;
    --cyan: #48d9e8;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #17274b 0%, transparent 42%),
        radial-gradient(ellipse at 90% 15%, #102a39 0%, transparent 35%),
        #080d19;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #edf3ff !important;
}

p, label, .stMarkdown {
    color: #c2cde0;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
}

.brand-icon {
    background: linear-gradient(135deg, #6c8cff, #42d7e8);
    color: #08101d;
    border-radius: 14px;
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 25px;
    font-weight: bold;
    box-shadow: 0 0 28px #4b80ff35;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #f4f7ff;
    letter-spacing: -0.8px;
}

.brand-sub {
    color: #8fa2c2;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero {
    padding: 30px 0 26px 0;
}

.eyebrow {
    color: #69ddeb;
    font-size: 12px;
    letter-spacing: 2.5px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.hero h1 {
    font-size: clamp(34px, 5vw, 57px);
    line-height: 1.08;
    letter-spacing: -2px;
    margin: 0 0 18px 0;
}

.gradient-text {
    background: linear-gradient(90deg, #83a1ff, #56e2e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #9cabc3;
    font-size: 16px;
    max-width: 680px;
    line-height: 1.8;
}

.status-pill {
    display: inline-block;
    border: 1px solid #29475a;
    border-radius: 30px;
    padding: 8px 13px;
    background: #10232d;
    color: #75e4d1;
    font-size: 12px;
    margin: 12px 0 25px 0;
}

.section-label {
    color: #edf3ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px;
    font-weight: 600;
    margin: 12px 0 5px 0;
}

.section-sub {
    color: #8fa0ba;
    font-size: 13px;
    margin-bottom: 22px;
}

.panel {
    background: linear-gradient(145deg, #121d31, #0d1525);
    border: 1px solid #27354e;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 14px 40px #00000017;
}

.panel-title {
    color: #edf3ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 5px;
}

.panel-caption {
    color: #8fa0ba;
    font-size: 12px;
    margin-bottom: 18px;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background-color: #0a1221 !important;
    color: #edf3ff !important;
    border: 1px solid #2b3b56 !important;
    border-radius: 10px !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #718fff !important;
    box-shadow: 0 0 0 1px #718fff !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label {
    color: #c5d2e7 !important;
    font-weight: 600 !important;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 48px;
    border: 0;
    border-radius: 11px;
    background: linear-gradient(100deg, #718cff, #47d5e3);
    color: #07111e !important;
    font-weight: 700;
    font-size: 15px;
    transition: 0.2s ease;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 0 28px #688dff48;
    transform: translateY(-1px);
}

.result-card {
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
    border: 1px solid #30415c;
    background: linear-gradient(145deg, #131f34, #0d1626);
}

.result-label {
    color: #9aacc8;
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
    text-transform: uppercase;
}

.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    margin: 12px 0;
}

.result-note {
    color: #a8b7ce;
    font-size: 13px;
    line-height: 1.7;
}

.footer {
    border-top: 1px solid #253149;
    margin-top: 55px;
    padding-top: 20px;
    color: #71829e;
    font-size: 12px;
    text-align: center;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

@media (max-width: 700px) {
    .hero h1 { letter-spacing: -1px; }
    .panel { padding: 16px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    model = joblib.load("fake_job_detector.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    columns = joblib.load("structured_columns.pkl")
    return model, tfidf, columns


# ---------------- HEADER ----------------
st.markdown("""
<div class="brand">
    <div class="brand-icon">V</div>
    <div>
        <div class="brand-name">VeriJob AI</div>
        <div class="brand-sub">Intelligent job screening</div>
    </div>
</div>

<div class="hero">
    <div class="eyebrow">✦ AI-Powered Fraud Detection</div>
    <h1>Trust the opportunity.<br>
    <span class="gradient-text">Verify the job.</span></h1>
    <p>
        Analyze job posting details with a machine-learning model
        designed to identify potentially fraudulent recruitment offers.
        Make more informed decisions before you apply.
    </p>
    <div class="status-pill">● &nbsp; Job analysis workspace</div>
</div>
""", unsafe_allow_html=True)


# ---------------- INPUT SECTION ----------------
st.markdown('<div class="section-label">Analyze a job posting</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Enter the available details below to begin your analysis.</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1], gap="large")

with st.form("job_analysis_form"):
    with left:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">01 / Job information</div>
            <div class="panel-caption">Basic details about the opportunity</div>
        </div>
        """, unsafe_allow_html=True)

        title = st.text_input(
            "Job title *",
            placeholder="e.g. Software Engineer"
        )
        company_profile = st.text_area(
            "Company profile",
            placeholder="What does the company do?",
            height=135
        )
        benefits = st.text_area(
            "Job benefits",
            placeholder="Salary, insurance, paid leave, etc.",
            height=110
        )

    with right:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">02 / Role details</div>
            <div class="panel-caption">Responsibilities and qualifications</div>
        </div>
        """, unsafe_allow_html=True)

        description = st.text_area(
            "Job description *",
            placeholder="Describe the responsibilities and duties...",
            height=190
        )
        requirements = st.text_area(
            "Job requirements",
            placeholder="Skills, qualifications, and experience...",
            height=165
        )

    submitted = st.form_submit_button(
        "✦  Analyze Job Posting",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:
    if not title.strip() or not description.strip():
        st.warning("Please fill in the job title and job description.")
    else:
        try:
            with st.spinner("Analyzing job posting..."):
                model, tfidf, structured_columns = load_files()

                text = " ".join([
                    title,
                    company_profile,
                    description,
                    requirements,
                    benefits
                ])

                # Keep this simple preprocessing for the current app version.
                text = text.lower()
                text = re.sub(r"[^a-z\s]", " ", text)
                text = re.sub(r"\s+", " ", text).strip()

                text_features = tfidf.transform([text])

                structured_data = pd.DataFrame(
                    [[0] * len(structured_columns)],
                    columns=structured_columns
                ).astype(float)

                all_features = hstack([
                    text_features,
                    structured_data.values
                ])

                prediction = model.predict(all_features)[0]

            st.markdown("---")
            st.markdown(
                '<div class="section-label">Analysis result</div>',
                unsafe_allow_html=True
            )

            if prediction == 1:
                st.markdown("""
                <div class="result-card" style="border-color:#753c4c;">
                    <div class="result-label">Prediction outcome</div>
                    <div class="result-title" style="color:#ff8998;">
                        ⚠ Potentially fraudulent
                    </div>
                    <div class="result-note">
                        The model classified this posting as potentially fake.
                        Verify the employer, check the official company website,
                        and never pay a fee to secure a job.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card" style="border-color:#286657;">
                    <div class="result-label">Prediction outcome</div>
                    <div class="result-title" style="color:#68e1b5;">
                        ✓ Classified as genuine
                    </div>
                    <div class="result-note">
                        The model classified this posting as likely genuine.
                        This does not confirm that the job is legitimate.
                        Verify the employer and offer independently.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "AI predictions can be incorrect. This result is informational "
                "and should not be treated as proof of legitimacy."
            )

        except FileNotFoundError:
            st.error(
                "A model file is missing. Make sure all three .pkl files "
                "are in the same folder as app.py."
            )
        except Exception as e:
            st.error("The analysis could not be completed.")
            st.code(str(e))


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    VERIJOB AI &nbsp; • &nbsp; MACHINE-LEARNING JOB SCREENING<br><br>
    Built for safer job searches · Always verify before you trust.
</div>
""", unsafe_allow_html=True)