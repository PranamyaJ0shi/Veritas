"""
Streamlit Web Application for Fake News Detection.
Provides interactive article verification, online URL extraction, confidence scoring,
linguistic feature breakdown, explainability cues, and model benchmarking.
"""

import sys
import os
import json
import streamlit as st
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_article
from src.url_extractor import extract_article_from_url

# Page configuration
st.set_page_config(
    page_title="Veritas | Fake News & Credibility Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, bold styling and high-contrast accessibility
st.markdown("""
<style>
    /* Main Hero Headers */
    .main-header {
        font-size: 2.85rem;
        font-weight: 950;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        line-height: 1.15;
        margin-bottom: 0.4rem;
    }
    .sub-header {
        font-size: 1.22rem;
        font-weight: 500;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 1.8rem;
    }

    /* Tabs Styling - Bigger and Bolder */
    button[data-baseweb="tab"] {
        font-size: 1.18rem !important;
        font-weight: 750 !important;
        padding: 12px 24px !important;
        color: #E2E8F0 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60A5FA !important;
    }

    /* Form Labels & Controls */
    [data-testid="stWidgetLabel"] p {
        font-size: 1.18rem !important;
        font-weight: 800 !important;
        color: #F1F5F9 !important;
        margin-bottom: 0.3rem !important;
    }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        font-size: 1.08rem !important;
        line-height: 1.5 !important;
    }

    /* Big Bold Primary Buttons */
    button[kind="primary"] {
        font-size: 1.25rem !important;
        font-weight: 850 !important;
        padding: 14px 28px !important;
        border-radius: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6) !important;
    }

    /* Big Bold Verdict Hero Cards */
    .verdict-hero-card {
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        border-width: 2px;
        border-style: solid;
    }
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.28) 100%);
        border-color: #EF4444;
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.18) 0%, rgba(21, 128, 61, 0.28) 100%);
        border-color: #22C55E;
    }
    .verdict-tag {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .tag-fake { color: #FCA5A5; }
    .tag-real { color: #86EFAC; }

    .verdict-title {
        font-size: 2.4rem;
        font-weight: 950;
        line-height: 1.1;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }
    .title-fake { color: #F87171; }
    .title-real { color: #4ADE80; }

    .verdict-desc {
        font-size: 1.05rem;
        font-weight: 500;
        color: #E2E8F0;
    }

    /* Big Bold Streamlit Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 900 !important;
        color: #F8FAFC !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.05rem !important;
        font-weight: 750 !important;
        color: #94A3B8 !important;
        letter-spacing: 0.4px !important;
    }

    /* Glowing Cue Badges */
    .cue-pill {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 24px;
        font-size: 1.02rem;
        font-weight: 750;
        margin: 4px 6px 4px 0;
        border-width: 1px;
        border-style: solid;
    }
    .cue-fake {
        background-color: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border-color: #EF4444;
    }
    .cue-real {
        background-color: rgba(34, 197, 94, 0.18);
        color: #86EFAC;
        border-color: #22C55E;
    }
    .cue-satire {
        background-color: rgba(234, 179, 8, 0.22);
        color: #FDE047;
        border-color: #EAB308;
    }

    /* Disclaimer Box - Bolder */
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.14);
        border-left: 5px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 2rem;
        color: #FDE68A;
        font-size: 1.02rem;
        font-weight: 500;
        line-height: 1.6;
    }
    .disclaimer-box strong {
        font-size: 1.1rem;
        color: #FBBF24;
    }

    /* Section Headers */
    .section-title {
        font-size: 1.65rem;
        font-weight: 850;
        color: #F8FAFC;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
        letter-spacing: -0.3px;
    }
</style>
""", unsafe_allow_html=True)

# Sample articles for quick testing
SAMPLE_ARTICLES = {
    "Select an example...": {
        "title": "",
        "text": ""
    },
    "Example 1: Reliable Science News (James Webb Telescope)": {
        "title": "NASA James Webb Space Telescope Observes Atmospheric Compounds on Exoplanet",
        "text": "Astronomers analyzing spectroscopic data collected by NASA's James Webb Space Telescope have identified water vapor, sulfur dioxide, and carbon dioxide in the atmosphere of an exoplanet located 700 light-years away from Earth. The findings, published in the journal Nature, provide unprecedented insight into the chemical composition of giant gas planets orbiting distant stars. 'These observations demonstrate the transformative capability of the observatory,' stated Dr. Natalie Batalha, lead investigator from the University of California. The research team emphasized that while the planet cannot support life due to extreme surface temperatures exceeding 1,000 degrees Celsius, the methodology paves the way for studying smaller, potentially habitable rocky planets."
    },
    "Example 2: Reliable Economic Report (Federal Reserve)": {
        "title": "Federal Reserve Holds Interest Rates Steady Amid Cooling Inflation Data",
        "text": "The Federal Reserve decided on Wednesday to keep its benchmark interest rate unchanged in the range of 5.25% to 5.50%. Federal Reserve Chairman Jerome Powell noted during the post-meeting press conference that while inflation has decelerated significantly over the past six months, policymakers require more sustained evidence before reducing borrowing costs. According to the Bureau of Labor Statistics, the Consumer Price Index rose by 3.1 percent on an annualized basis last month, aligning with economic consensus estimates. Financial market analysts from Goldman Sachs expect rate cuts could begin later in the year, provided employment figures remain resilient."
    },
    "Example 3: Sensational Fake News (Secret Miracle Cure)": {
        "title": "Miracle Kitchen Spice Completely CURES All Stages Of Cancer in 24 Hours, Big Pharma Panicking!",
        "text": "A revolutionary cure for all forms of cancer has been deliberately hidden from the public by corrupt pharmaceutical executives who make trillions off chemotherapy treatments! A top secret laboratory discovered that common kitchen turmeric mixed with crushed apple seeds destroys 100% of malignant cancer cells within exactly 24 hours with ZERO side effects! Doctors are terrified of losing their medical licenses if they prescribe this natural miracle remedy! Greedy billion-dollar hospital conglomerates are suing to suppress the truth, but insiders have finally leaked the natural cure recipe! Do not let Big Pharma poison your family, drink this every morning!"
    },
    "Example 4: Sensational Conspiracy (Secret Microchips)": {
        "title": "SHOCKING SECRET: Secret World Government Will FORCE Everyone Into Digital Microchips By Midnight!",
        "text": "YOU WON'T BELIEVE WHAT INSIDERS JUST LEAKED! Corrupt globalist elites have secretly signed a clandestine treaty in Switzerland to eliminate all paper cash and FORCE every single human being on Earth to be implanted with biometric microchips before midnight! Mainstream media is completely SILENT because they are controlled by the shadow regime! Whistleblowers who tried to speak out have disappeared overnight! If you refuse the chip, your bank accounts will be confiscated immediately and you will be barred from buying groceries! SHARE THIS EVERYWHERE BEFORE IT GETS CENSORED AND PULLED DOWN FOREVER! WAKE UP PEOPLE!"
    }
}

# Session state initialization for extracted or loaded content
if "input_title" not in st.session_state:
    st.session_state["input_title"] = ""
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

# Sidebar navigation & branding
with st.sidebar:
    st.image("https://img.icons8.com/color/96/news.png", width=72)
    st.markdown("""
    <h1 style="font-size: 2.3rem; font-weight: 900; color: var(--text-color, #F8FAFC); margin-top: 0.3rem; margin-bottom: 0.2rem; line-height: 1.1; letter-spacing: -0.5px;">
        VERITAS CLASSIFIER
    </h1>
    <p style="font-size: 0.95rem; font-weight: 600; color: #60A5FA; margin-bottom: 1rem;">
        Misinformation & Stylometry Intelligence
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Veritas Classifier** is an automated credibility analytics engine that inspects digital news text using Natural Language Processing and stylometric signal modeling. It evaluates sensationalism, readability, attribution cues, and deceptive lexical signatures.
    """)

    st.markdown("---")
    st.markdown("### 🌐 Market & Industry Uses")
    st.markdown("""
    - **🛡️ Social Platforms & Moderation:** Automates early detection of sensationalist clickbait and viral disinformation before amplification.
    - **📰 Digital Journalism & Newsrooms:** Equips fact-checkers and reporters with automated credibility scoring for user-submitted leads and breaking news.
    - **🏛️ Brand Safety & Enterprise PR:** Protects corporate reputation by identifying fake PR wires, executive impersonation stories, and smear campaigns.
    - **📈 Financial & Market Intelligence:** Screens financial newsfeeds and algorithmic trading pipelines against artificial rumors designed to trigger market volatility.
    - **🎯 Ad-Tech & Content Discovery:** Enables programmatic ad networks and search feeds to penalize low-credibility domains, preserving ad quality and brand safety.
    """)

# Main header - Bigger and Bolder
st.markdown('<div class="main-header">VERITAS • FAKE NEWS CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluating digital articles through natural language processing, linguistic stylometry, and credibility cue detection.</div>', unsafe_allow_html=True)

def render_analysis_results(result):
    st.markdown("---")
    
    # Hero Verdict Card + Confidence Meter
    res_col1, res_col2 = st.columns([3, 2])
    
    with res_col1:
        if result.get("is_satire", False):
            st.markdown(f"""
            <div class="verdict-hero-card verdict-fake">
                <div class="verdict-tag tag-fake">🎭 CLASSIFICATION VERDICT: SATIRICAL PARODY</div>
                <div class="verdict-title title-fake">LIKELY UNRELIABLE</div>
                <div class="verdict-desc">Article originates from a known satirical publication or exhibits deadpan parody styling. While grammar and sentence length mimic professional journalism, the premises are fictional satire.</div>
            </div>
            """, unsafe_allow_html=True)
        elif result["is_fake"]:
            st.markdown(f"""
            <div class="verdict-hero-card verdict-fake">
                <div class="verdict-tag tag-fake">⚠️ CLASSIFICATION VERDICT</div>
                <div class="verdict-title title-fake">LIKELY UNRELIABLE</div>
                <div class="verdict-desc">High presence of sensationalist tone, clickbait triggers, emotional appeals, or lack of factual attribution.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-hero-card verdict-real">
                <div class="verdict-tag tag-real">✅ CLASSIFICATION VERDICT</div>
                <div class="verdict-title title-real">LIKELY RELIABLE</div>
                <div class="verdict-desc">Text exhibits neutral journalistic phrasing, verifiable attribution cues, and standard readability metrics.</div>
            </div>
            """, unsafe_allow_html=True)
    
    with res_col2:
        st.metric(
            label="MODEL CONFIDENCE SCORE",
            value=f"{result['confidence_percentage']}%",
            delta=f"{result['label']}"
        )
        st.write("**Unreliability Index Gauge:**")
        st.progress(float(result["prob_fake"]), text=f"Unreliability Score: {int(result['prob_fake'] * 100)}%")
        st.caption("Probability calculated by hybrid TF-IDF sublinear vectors and 18 scaled linguistic features.")

    # Explainability & Cues Section
    st.markdown('<div class="section-title">🔍 Key Detected Signals & Explainability</div>', unsafe_allow_html=True)
    
    sat_cues = result["highlights"].get("satire_cues", [])
    if sat_cues:
        st.write("### 🎭 Satirical / Parody Publisher Identifiers")
        for cue in sat_cues:
            st.markdown(f'<span class="cue-pill cue-satire">🟡 SATIRE TRIGGER: {cue.upper()}</span>', unsafe_allow_html=True)
        st.caption("Deadpan satire deliberately mimics authentic journalistic grammar and neutral tone, while reporting fictional humorous events.")

    cues_col1, cues_col2 = st.columns(2)
    
    with cues_col1:
        st.write("### 🚨 Sensational / Clickbait Triggers Detected")
        sens_cues = result["highlights"]["sensational_cues"]
        if sens_cues:
            for cue in sens_cues:
                st.markdown(f'<span class="cue-pill cue-fake">🔴 {cue.upper()}</span>', unsafe_allow_html=True)
        else:
            st.markdown("*None detected. Writing style maintains neutral tone.*")

    with cues_col2:
        st.write("### 📰 Journalistic Attribution Cues Identified")
        attr_cues = result["highlights"]["attribution_cues"]
        if attr_cues:
            for cue in attr_cues:
                st.markdown(f'<span class="cue-pill cue-real">🟢 {cue.upper()}</span>', unsafe_allow_html=True)
        else:
            st.markdown("*No explicit journalistic attribution found.*")

    # Linguistic Stylometry Metrics
    st.markdown('<div class="section-title">📊 Linguistic Stylometry Breakdown</div>', unsafe_allow_html=True)
    metrics = result["linguistic_metrics"]
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Total Word Count", int(metrics["word_count"]))
        st.metric("Uppercase Ratio", f"{metrics['caps_ratio'] * 100:.1f}%")
    with m_col2:
        st.metric("Flesch Reading Ease", f"{metrics['flesch_reading_ease']:.1f}")
        st.metric("Exclamation Marks (!)", int(metrics["exclamation_count"]))
    with m_col3:
        st.metric("Gunning Fog Index", f"{metrics['gunning_fog_index']:.1f}")
        st.metric("Question Marks (?)", int(metrics["question_count"]))
    with m_col4:
        st.metric("Lexical Diversity (TTR)", f"{metrics['type_token_ratio']:.2f}")
        st.metric("Numerical Density", f"{metrics['numbers_ratio'] * 100:.1f}%")

    # Academic Disclaimer - Bolder
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Academic Disclaimer:</strong> This system uses Natural Language Processing and Machine Learning to analyze linguistic and stylistic patterns (such as sensational tone, emotional triggers, attribution cues, and readability). It does <em>not</em> verify objective real-time facts against primary sources. Always verify claims through reputable fact-checking organizations.
    </div>
    """, unsafe_allow_html=True)

# Main Analyzer View
input_mode = st.radio(
    "Choose Analysis Input Method:",
    ["🔗 Online Article URL (Automatic Web Extraction)", "📝 Paste Headline & Text (or Use Presets)"],
    horizontal=True
)

if input_mode.startswith("🔗"):
    st.markdown("### 🌐 Extract & Analyze Live Article from the Web")
    url_input = st.text_input(
        "Paste News Article URL:",
        placeholder="e.g. https://www.bbc.com/news/... or https://indianexpress.com/article/..."
    )
    fetch_btn = st.button("Fetch and Analyze URL", type="primary", use_container_width=True)

    if fetch_btn and url_input.strip():
        with st.spinner("Connecting to webpage and extracting article content..."):
            ext = extract_article_from_url(url_input)
            if ext["success"]:
                st.success(f"Successfully extracted: **{ext['title']}** ({len(ext['text'].split())} words)")
                with st.expander("View Extracted Article Content", expanded=False):
                    st.write(ext["text"])
                
                with st.spinner("Analyzing extracted text..."):
                    res = predict_article(ext["title"], ext["text"])
                    render_analysis_results(res)
            else:
                if ext.get("is_bot_blocked", False):
                    st.warning(f"🛡️ **Anti-Bot Protection / Paywall Detected:**\n\n{ext['error']}")
                    st.info("💡 **Quick Solution:** Switch to the **'📝 Paste Headline & Text'** option above and paste the article directly to analyze it.")
                else:
                    st.error(f"Could not extract article: {ext['error']}")
    elif fetch_btn:
        st.warning("Please enter a valid news URL.")

else:
    # Preset selection
    st.markdown("### 📝 Paste Article or Select a Preset")
    selected_preset = st.selectbox("Load Sample Benchmark Article:", list(SAMPLE_ARTICLES.keys()))
    if selected_preset != "Select an example...":
        current_title = SAMPLE_ARTICLES[selected_preset]["title"]
        current_text = SAMPLE_ARTICLES[selected_preset]["text"]
    else:
        current_title = st.session_state["input_title"]
        current_text = st.session_state["input_text"]

    with st.form("manual_form"):
        title_input = st.text_input(
            "Article Headline / Title:",
            value=current_title,
            placeholder="e.g. Federal Reserve Holds Interest Rates Steady..."
        )
        text_input = st.text_area(
            "Article Body Text:",
            value=current_text,
            height=220,
            placeholder="Paste the full article body text here..."
        )
        submit_btn = st.form_submit_button("Analyze Article Credibility", type="primary", use_container_width=True)

    if submit_btn:
        if not text_input.strip() and not title_input.strip():
            st.warning("Please provide a headline or article text to analyze.")
        else:
            with st.spinner("Extracting linguistic stylometry & computing credibility signals..."):
                try:
                    res = predict_article(title_input, text_input)
                    render_analysis_results(res)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
