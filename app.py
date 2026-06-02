# PROJECT: AI-Based Fake News Detection System

import streamlit as st
import pickle, re, nltk, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# Page Config
st.set_page_config(
    page_title="FakeShield AI | News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# MEGA CSS — 3D Modern Cyberpunk Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── CSS Variables ── */
:root {
  --cyan:    #00f5ff;
  --magenta: #ff006e;
  --purple:  #7b2fff;
  --dark:    #020408;
  --dark2:   #050d1a;
  --dark3:   #081020;
  --glass:   rgba(0,245,255,0.03);
  --border:  rgba(0,245,255,0.15);
  --glow:    0 0 20px rgba(0,245,255,0.3);
  --glow2:   0 0 40px rgba(255,0,110,0.3);
}

/* ── Global Reset ── */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── App Background with Animated Grid ── */
.stApp {
  background: var(--dark) !important;
  font-family: 'Rajdhani', sans-serif !important;
  background-image:
    linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
  0%   { background-position: 0 0; }
  100% { background-position: 50px 50px; }
}

/* ── Animated Background Orbs ── */
.stApp::before {
  content: '';
  position: fixed;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background:
    radial-gradient(ellipse at 20% 20%, rgba(123,47,255,0.12) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(255,0,110,0.08) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(0,245,255,0.05) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
  animation: orbMove 15s ease-in-out infinite alternate;
}

@keyframes orbMove {
  0%   { transform: translate(0,0) rotate(0deg); }
  100% { transform: translate(30px, 20px) rotate(5deg); }
}

.block-container {
  padding: 1.5rem 2.5rem !important;
  max-width: 1300px !important;
  position: relative;
  z-index: 1;
}

/* ══════════════════════════════════
   HERO SECTION
══════════════════════════════════ */
.hero-wrap {
  position: relative;
  background: linear-gradient(135deg, rgba(0,245,255,0.05), rgba(123,47,255,0.08), rgba(255,0,110,0.05));
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 3rem 2rem 2.5rem;
  margin-bottom: 2rem;
  text-align: center;
  overflow: hidden;
  backdrop-filter: blur(10px);
  box-shadow: var(--glow), inset 0 1px 0 rgba(0,245,255,0.1);
}

/* Scan line animation */
.hero-wrap::before {
  content: '';
  position: absolute;
  top: -100%;
  left: 0;
  width: 100%;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation: scan 4s linear infinite;
}

.hero-wrap::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,245,255,0.01) 2px,
    rgba(0,245,255,0.01) 4px
  );
  pointer-events: none;
}

@keyframes scan {
  0%   { top: -5%; }
  100% { top: 105%; }
}

.hero-badge {
  display: inline-block;
  background: rgba(0,245,255,0.1);
  border: 1px solid rgba(0,245,255,0.3);
  border-radius: 50px;
  padding: 0.3rem 1.2rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.75rem;
  color: var(--cyan);
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 1.2rem;
  animation: pulse-border 2s ease-in-out infinite;
}

@keyframes pulse-border {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,245,255,0.3); }
  50%       { box-shadow: 0 0 0 6px rgba(0,245,255,0); }
}

.hero-title {
  font-family: 'Orbitron', monospace;
  font-size: clamp(2rem, 5vw, 3.5rem);
  font-weight: 900;
  color: #ffffff;
  line-height: 1.1;
  letter-spacing: -1px;
  margin-bottom: 0.8rem;
  text-shadow: 0 0 30px rgba(0,245,255,0.5), 0 0 60px rgba(0,245,255,0.2);
}

.hero-title .accent1 { color: var(--cyan); }
.hero-title .accent2 {
  background: linear-gradient(135deg, var(--magenta), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.9rem;
  color: rgba(0,245,255,0.6);
  letter-spacing: 2px;
  margin-bottom: 2rem;
}

/* Stat Pills */
.stat-row {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 1.5rem;
}

.stat-pill {
  background: rgba(0,0,0,0.4);
  border: 1px solid rgba(0,245,255,0.2);
  border-radius: 50px;
  padding: 0.5rem 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  backdrop-filter: blur(5px);
}

.stat-pill .num {
  font-family: 'Orbitron', monospace;
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--cyan);
}

.stat-pill .lbl {
  font-size: 0.8rem;
  color: rgba(255,255,255,0.5);
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* ══════════════════════════════════
   3D CARDS — Glassmorphism
══════════════════════════════════ */
.glass-card {
  background: linear-gradient(135deg, rgba(0,245,255,0.04), rgba(0,0,0,0.6));
  border: 1px solid rgba(0,245,255,0.12);
  border-radius: 16px;
  padding: 1.8rem;
  backdrop-filter: blur(20px);
  position: relative;
  overflow: hidden;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  transform-style: preserve-3d;
}

.glass-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.5;
}

.glass-card:hover {
  transform: translateY(-4px) perspective(500px) rotateX(1deg);
  box-shadow: 0 20px 60px rgba(0,0,0,0.5), var(--glow);
}

/* ══════════════════════════════════
   RESULT CARDS — 3D Effect
══════════════════════════════════ */
.result-card {
  border-radius: 20px;
  padding: 2.5rem 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  transform-style: preserve-3d;
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  animation: cardAppear 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

@keyframes cardAppear {
  0%   { opacity: 0; transform: scale(0.8) translateY(30px); }
  100% { opacity: 1; transform: scale(1) translateY(0); }
}

.result-card:hover {
  transform: translateY(-8px) perspective(600px) rotateX(3deg);
}

.result-card-fake {
  background: linear-gradient(135deg, rgba(255,0,110,0.12), rgba(255,0,110,0.05), rgba(0,0,0,0.7));
  border: 2px solid rgba(255,0,110,0.5);
  box-shadow: 0 0 40px rgba(255,0,110,0.2), inset 0 1px 0 rgba(255,0,110,0.2);
}

.result-card-real {
  background: linear-gradient(135deg, rgba(0,245,255,0.12), rgba(0,245,255,0.05), rgba(0,0,0,0.7));
  border: 2px solid rgba(0,245,255,0.5);
  box-shadow: 0 0 40px rgba(0,245,255,0.2), inset 0 1px 0 rgba(0,245,255,0.2);
}

/* Corner accent on result cards */
.result-card::before {
  content: '';
  position: absolute;
  top: 0; right: 0;
  width: 60px; height: 60px;
  border-top: 2px solid; border-right: 2px solid;
  border-radius: 0 20px 0 0;
}
.result-card-fake::before { border-color: rgba(255,0,110,0.5); }
.result-card-real::before { border-color: rgba(0,245,255,0.5); }

.result-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0;
  width: 60px; height: 60px;
  border-bottom: 2px solid; border-left: 2px solid;
  border-radius: 0 0 0 20px;
}
.result-card-fake::after { border-color: rgba(255,0,110,0.5); }
.result-card-real::after { border-color: rgba(0,245,255,0.5); }

.res-icon { font-size: 4rem; line-height: 1; margin-bottom: 0.8rem; display: block; animation: iconPop 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) 0.3s both; }
@keyframes iconPop { 0% { transform: scale(0); } 100% { transform: scale(1); } }

.res-model { font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: rgba(255,255,255,0.4); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem; }
.res-verdict { font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 900; letter-spacing: 2px; margin-bottom: 0.5rem; }
.res-verdict-fake { color: #ff006e; text-shadow: 0 0 20px rgba(255,0,110,0.6); }
.res-verdict-real { color: #00f5ff; text-shadow: 0 0 20px rgba(0,245,255,0.6); }
.res-confidence { font-size: 1rem; color: rgba(255,255,255,0.5); font-family: 'Share Tech Mono', monospace; }

/* Confidence progress bar */
.conf-bar-wrap { margin-top: 1.2rem; background: rgba(0,0,0,0.4); border-radius: 50px; height: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
.conf-bar-fill-fake { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #ff006e, #ff4da6); animation: fillBar 1s ease 0.5s both; }
.conf-bar-fill-real { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #00b4d8, #00f5ff); animation: fillBar 1s ease 0.5s both; }
@keyframes fillBar { from { width: 0%; } }

/* ══════════════════════════════════
   INPUT TEXTAREA
══════════════════════════════════ */
.stTextArea > div > div > textarea {
  background: rgba(0,10,20,0.8) !important;
  border: 1px solid rgba(0,245,255,0.2) !important;
  border-radius: 12px !important;
  color: rgba(255,255,255,0.9) !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1rem !important;
  padding: 1rem !important;
  transition: all 0.3s ease !important;
  resize: vertical !important;
}
.stTextArea > div > div > textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,245,255,0.1), var(--glow) !important;
  outline: none !important;
}
.stTextArea > div > div > textarea::placeholder { color: rgba(0,245,255,0.25) !important; font-style: italic; }

/* ══════════════════════════════════
   BUTTONS
══════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(123,47,255,0.15)) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0,245,255,0.4) !important;
  border-radius: 10px !important;
  padding: 0.75rem 2rem !important;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  width: 100% !important;
  position: relative !important;
  overflow: hidden !important;
  transition: all 0.3s ease !important;
  backdrop-filter: blur(10px) !important;
}

.stButton > button::before {
  content: '' !important;
  position: absolute !important;
  top: 0; left: -100% !important;
  width: 100%; height: 100% !important;
  background: linear-gradient(90deg, transparent, rgba(0,245,255,0.1), transparent) !important;
  transition: left 0.5s ease !important;
}

.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,245,255,0.25), rgba(123,47,255,0.25)) !important;
  box-shadow: var(--glow), 0 8px 30px rgba(0,0,0,0.4) !important;
  transform: translateY(-2px) !important;
  border-color: var(--cyan) !important;
}

.stButton > button:hover::before { left: 100% !important; }

.stButton > button:active { transform: translateY(0) !important; }

/* Primary analyze button — bigger glow */
div[data-testid="column"]:has(.main-btn) .stButton > button,
.analyze-btn .stButton > button {
  background: linear-gradient(135deg, #00f5ff22, #7b2fff22, #ff006e22) !important;
  border: 1px solid var(--cyan) !important;
  font-size: 0.9rem !important;
  padding: 1rem !important;
  animation: btnPulse 3s ease-in-out infinite !important;
}

@keyframes btnPulse {
  0%, 100% { box-shadow: 0 0 10px rgba(0,245,255,0.2); }
  50%       { box-shadow: 0 0 30px rgba(0,245,255,0.5), 0 0 60px rgba(0,245,255,0.1); }
}

/* ══════════════════════════════════
   SELECTBOX
══════════════════════════════════ */
.stSelectbox > div > div {
  background: rgba(0,10,20,0.8) !important;
  border: 1px solid rgba(0,245,255,0.2) !important;
  border-radius: 10px !important;
  color: white !important;
}
.stSelectbox > div > div:hover { border-color: var(--cyan) !important; }

/* ══════════════════════════════════
   SIDEBAR
══════════════════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #020d1a, #010810) !important;
  border-right: 1px solid rgba(0,245,255,0.1) !important;
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: 'Orbitron', monospace !important;
  color: var(--cyan) !important;
  font-size: 0.85rem !important;
  letter-spacing: 2px !important;
}
/* Sidebar reopen button */
[data-testid="collapsedControl"] {
  display: block !important;
  visibility: visible !important;
  background: rgba(0,245,255,0.1) !important;
  border: 1px solid rgba(0,245,255,0.3) !important;
  border-radius: 0 8px 8px 0 !important;
  top: 50% !important;
}
[data-testid="collapsedControl"]:hover {
  background: rgba(0,245,255,0.2) !important;
}

/* ══════════════════════════════════
   TABS
══════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(0,10,20,0.6) !important;
  border: 1px solid rgba(0,245,255,0.1) !important;
  border-radius: 12px !important;
  padding: 5px !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(255,255,255,0.4) !important;
  border-radius: 8px !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  letter-spacing: 1px !important;
  transition: all 0.3s ease !important;
  border: none !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--cyan) !important; background: rgba(0,245,255,0.05) !important; }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(123,47,255,0.15)) !important;
  color: var(--cyan) !important;
  border: 1px solid rgba(0,245,255,0.3) !important;
  box-shadow: var(--glow) !important;
}

/* ══════════════════════════════════
   EXPANDER
══════════════════════════════════ */
.streamlit-expanderHeader {
  background: rgba(0,245,255,0.05) !important;
  border: 1px solid rgba(0,245,255,0.1) !important;
  border-radius: 8px !important;
  color: var(--cyan) !important;
  font-family: 'Share Tech Mono', monospace !important;
}

/* ══════════════════════════════════
   LABELS + TEXT
══════════════════════════════════ */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: 'Orbitron', monospace !important;
  color: white !important;
}
.stMarkdown p, .stMarkdown li { color: rgba(255,255,255,0.75) !important; font-family: 'Rajdhani', sans-serif !important; font-size: 1rem !important; }
label, .stSelectbox label { color: rgba(0,245,255,0.7) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.78rem !important; letter-spacing: 1px !important; }

/* Section headings */
.sec-head {
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 3px;
  text-transform: uppercase;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(0,245,255,0.15);
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.8rem;
}
.sec-head::before {
  content: '';
  display: block;
  width: 4px;
  height: 20px;
  background: linear-gradient(180deg, var(--cyan), var(--purple));
  border-radius: 2px;
}

/* Info box */
.info-box {
  background: rgba(0,245,255,0.05);
  border: 1px solid rgba(0,245,255,0.15);
  border-left: 3px solid var(--cyan);
  border-radius: 0 10px 10px 0;
  padding: 0.8rem 1.2rem;
  color: rgba(0,245,255,0.8) !important;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.82rem;
  margin: 0.8rem 0;
}

/* Word/char counter */
.counter-text { font-family: 'Share Tech Mono', monospace !important; font-size: 0.78rem !important; color: rgba(0,245,255,0.4) !important; }

/* Table styling */
table { border-collapse: collapse !important; width: 100% !important; }
th { background: rgba(0,245,255,0.1) !important; color: var(--cyan) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.82rem !important; padding: 0.7rem !important; border: 1px solid rgba(0,245,255,0.1) !important; }
td { padding: 0.6rem !important; border: 1px solid rgba(0,245,255,0.07) !important; color: rgba(255,255,255,0.75) !important; font-family: 'Rajdhani', sans-serif !important; }
tr:hover td { background: rgba(0,245,255,0.03) !important; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
header { visibility: visible; background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* Metric styling */
[data-testid="metric-container"] {
  background: rgba(0,245,255,0.04) !important;
  border: 1px solid rgba(0,245,255,0.1) !important;
  border-radius: 12px !important;
  padding: 1rem !important;
}

/* Alert/info boxes from streamlit */
.stAlert { border-radius: 10px !important; border-left: 3px solid var(--cyan) !important; background: rgba(0,245,255,0.05) !important; }
.stWarning { border-left-color: #ffb700 !important; background: rgba(255,183,0,0.05) !important; }
.stError { border-left-color: var(--magenta) !important; background: rgba(255,0,110,0.05) !important; }

/* Code block */
code { background: rgba(0,245,255,0.08) !important; color: var(--cyan) !important; border: 1px solid rgba(0,245,255,0.15) !important; border-radius: 5px !important; font-family: 'Share Tech Mono', monospace !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--cyan) !important; }
</style>
""", unsafe_allow_html=True)
# BACKEND: Load Models

@st.cache_resource
def load_models():
    with open("lr_model.pkl","rb") as f: lr  = pickle.load(f)
    with open("nb_model.pkl","rb") as f: nb  = pickle.load(f)
    with open("tfidf.pkl",   "rb") as f: tfi = pickle.load(f)
    return lr, nb, tfi

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+|www\S+','',str(text))
    text = re.sub(r'[^a-zA-Z\s]','',text)
    text = text.lower()
    words = [lemmatizer.lemmatize(w) for w in text.split()
             if w not in stop_words and len(w) > 2]
    return " ".join(words)

def predict(text, model, tfidf):
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    return pred, proba, cleaned
# HERO BANNER

st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">⚡ AI-POWERED DETECTION SYSTEM</div>
  <div class="hero-title">
    <span class="accent1">FAKE</span>SHIELD<br>
    <span class="accent2">NEWS INTELLIGENCE</span>
  </div>
  <div class="hero-subtitle">MACHINE LEARNING • NLP • REAL-TIME ANALYSIS</div>
  <div class="stat-row">
    <div class="stat-pill"><span class="num">44K+</span><span class="lbl">Trained Articles</span></div>
    <div class="stat-pill"><span class="num">98%</span><span class="lbl">Accuracy</span></div>
    <div class="stat-pill"><span class="num">2</span><span class="lbl">AI Models</span></div>
    <div class="stat-pill"><span class="num">⚡</span><span class="lbl">Real-Time</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
# SIDEBAR

with st.sidebar:
    st.markdown("## 🛡️ FAKESHIELD")
    st.markdown("---")

    model_choice = st.selectbox(
        "SELECT AI MODEL",
        ["⚡ Logistic Regression", "🧠 Naive Bayes", "🔥 Both (Compare)"],
    )

    st.markdown("---")
    st.markdown("### ⚡ QUICK TEST")
    st.caption("Load sample articles:")

    FAKE = """BREAKING: Secret government documents have confirmed that 5G towers are
    transmitting mind control frequencies that make people obedient to the global elite.
    Whistleblowers from inside the CIA have leaked proof that Bill Gates personally
    funded this program through a secret agreement with extraterrestrial entities.
    The mainstream media is suppressing this truth to protect their corporate masters."""

    REAL = """The Federal Reserve raised its benchmark interest rate by a quarter percentage
    point on Wednesday, bringing borrowing costs to their highest level in 22 years as
    central bank officials continued their campaign against inflation. Fed Chair Jerome
    Powell said the labor market remains strong and the committee remains committed to
    returning inflation to its 2 percent target. The decision was unanimous."""

    if st.button("📰 Load FAKE Example"):
        st.session_state["text"] = FAKE
        st.rerun()
    if st.button("📰 Load REAL Example"):
        st.session_state["text"] = REAL
        st.rerun()
    if st.button("🗑️ Clear Input"):
        st.session_state["text"] = ""
        st.rerun()

    st.markdown("---")
    st.markdown("### 📡 SYSTEM INFO")
    st.markdown("""
    **Models:** LR + Naive Bayes  
    **NLP:** NLTK Lemmatizer  
    **Features:** TF-IDF (50K)  
    **Dataset:** 44,898 articles  
    """)
    st.markdown("---")
    st.markdown("### 🎓 PROJECT INFO")
    st.markdown("""
    **BS Computer Science**  
    Final Year Project  
    *AI & ML Department*
    """)
# MAIN TABS

tab1, tab2, tab3 = st.tabs([
    "    ANALYZE NEWS  ",
    "    PERFORMANCE   ",
    "    DOCUMENTATION "
])
# TAB 1: ANALYZE

with tab1:
    try:
        lr_model, nb_model, tfidf = load_models()
        ready = True
    except FileNotFoundError:
        st.error("⚠️ Models not found! Run: python train_model.py")
        ready = False

    # Input Section
    st.markdown('<div class="sec-head">INPUT ARTICLE</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Paste the complete news article below. More text = higher accuracy. Minimum 10 words recommended.</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "ARTICLE TEXT INPUT",
        value=st.session_state.get("text",""),
        height=200,
        placeholder="[ PASTE NEWS ARTICLE HERE... ]\n\nSupports: Headlines • Full articles • Paragraphs",
        label_visibility="visible"
    )

    # Stats row
    words = len(user_input.split()) if user_input.strip() else 0
    chars = len(user_input)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("WORDS",  f"{words:,}")
    with c2: st.metric("CHARS",  f"{chars:,}")
    with c3: st.metric("MODEL",  model_choice.split()[1] if len(model_choice.split())>1 else "LR")
    with c4: st.metric("STATUS", "READY" if ready else "NO MODEL")

    st.markdown("<br>", unsafe_allow_html=True)

    clicked = st.button("⚡  ANALYZE THIS ARTICLE  ⚡", use_container_width=True)

    # ── Process ──
    if clicked:
        if not user_input.strip():
            st.warning(" INPUT REQUIRED — Please paste a news article.")
        elif words < 5:
            st.warning("TOO SHORT — Please enter at least 10 words for accurate results.")
        elif not ready:
            st.error(" Run python train_model.py first!")
        else:
            with st.spinner("ANALYZING ARTICLE..."):
                if "Both" in model_choice:
                    to_use = [("Logistic Regression", lr_model), ("Naive Bayes", nb_model)]
                elif "Logistic" in model_choice:
                    to_use = [("Logistic Regression", lr_model)]
                else:
                    to_use = [("Naive Bayes", nb_model)]

                results = []
                for name, mdl in to_use:
                    p, prob, cln = predict(user_input, mdl, tfidf)
                    results.append((name, p, prob, cln))

            # ── Results ──
            st.markdown("---")
            st.markdown('<div class="sec-head">DETECTION RESULTS</div>', unsafe_allow_html=True)

            cols = st.columns(len(results))
            for col, (name, pred, proba, cleaned) in zip(cols, results):
                with col:
                    if pred == 1:
                        conf=proba[1]*100; ctype="fake"; icon="❌"; verdict="FAKE NEWS"; bar_class="conf-bar-fill-fake"
                    else:
                        conf=proba[0]*100; ctype="real"; icon="✅"; verdict="REAL NEWS"; bar_class="conf-bar-fill-real"

                    st.markdown(f"""
                    <div class="result-card result-card-{ctype}">
                      <span class="res-icon">{icon}</span>
                      <div class="res-model">{name}</div>
                      <div class="res-verdict res-verdict-{ctype}">{verdict}</div>
                      <div class="res-confidence">CONFIDENCE: {conf:.1f}%</div>
                      <div class="conf-bar-wrap">
                        <div class="{bar_class}" style="width:{conf}%"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Probability Chart
                    st.markdown("<br>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(4.5, 3))
                    fig.patch.set_facecolor('#020d1a')
                    ax.set_facecolor('#020d1a')

                    colors = ['#00f5ff', '#ff006e']
                    vals   = [proba[0]*100, proba[1]*100]
                    bars   = ax.bar(["REAL", "FAKE"], vals, color=colors,
                                    width=0.5, edgecolor='none',
                                    linewidth=0)

                    # Glow effect — draw bars twice with alpha
                    for b, c in zip(bars, colors):
                     h = b.get_height()
                    ax.bar(b.get_x() + b.get_width()/2, h,
                          color=c, alpha=0.15, width=0.7)
                    ax.text(b.get_x()+b.get_width()/2, h+2,
                                f"{h:.1f}%", ha='center',
                                fontsize=11, color='white',
                                fontweight='bold',
                                fontfamily='monospace')

                    ax.set_ylim(0, 118)
                    ax.set_title("PROBABILITY ANALYSIS", color='#00f5ff',
                                 fontsize=9, pad=10, fontfamily='monospace',
                                 fontweight='bold')

                    ax.tick_params(colors='#ffffff60', labelsize=9)
                    for label in ax.get_xticklabels():
                        label.set_fontfamily('monospace')
                        label.set_color('#ffffff80')
                    ax.set_ylabel("SCORE (%)", color='#00f5ff60',
                                  fontsize=8, fontfamily='monospace')

                    for spine in ['top','right']:
                        ax.spines[spine].set_visible(False)
                    for spine in ['bottom','left']:
                        ax.spines[spine].set_color('#ffffff15')

                    ax.yaxis.grid(True, color='#ffffff08', linewidth=0.5)
                    ax.set_axisbelow(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            # ── Word Cloud ──
            st.markdown("---")
            st.markdown('<div class="sec-head">WORD FREQUENCY ANALYSIS</div>', unsafe_allow_html=True)
            cleaned_text = results[0][3]
            if cleaned_text.strip():
                wc = WordCloud(
                    width=1000, height=320,
                    background_color='#020408',
                    colormap='cool',
                    max_words=80,
                    prefer_horizontal=0.8,
                    collocations=False
                ).generate(cleaned_text)

                fig, ax = plt.subplots(figsize=(12, 4))
                fig.patch.set_facecolor('#020408')
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Not enough content after preprocessing to generate word cloud.")

            # ── Preprocessed Text ──
            with st.expander("🔬 VIEW PREPROCESSED TEXT (After NLP Pipeline)"):
                st.markdown(f"""
                <div class="info-box">
                ORIGINAL: {words} words → AFTER CLEANING: {len(cleaned_text.split())} words
                </div>
                """, unsafe_allow_html=True)
                st.code(cleaned_text, language=None)
# TAB 2: PERFORMANCE

with tab2:
    st.markdown('<div class="sec-head">MODEL PERFORMANCE DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📊 These visualizations are auto-generated when you run python train_model.py</div>', unsafe_allow_html=True)

    try:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("** CONFUSION MATRICES**")
            st.caption("Correct vs incorrect predictions for each model")
            st.image("confusion_matrices.png", use_column_width=True)
        with c2:
            st.markdown("** MODEL ACCURACY COMPARISON**")
            st.caption("Accuracy, Precision, Recall, F1-Score")
            st.image("model_comparison.png", use_column_width=True)

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("** FAKE NEWS — WORD CLOUD**")
            st.caption("Most frequent words in fake articles")
            st.image("wordcloud_fake.png", use_column_width=True)
        with c4:
            st.markdown("** REAL NEWS — WORD CLOUD**")
            st.caption("Most frequent words in real articles")
            st.image("wordcloud_real.png", use_column_width=True)

    except Exception:
        st.warning(" Plots not found. Run: python train_model.py")

    # Metrics explanation
    st.markdown("---")
    st.markdown('<div class="sec-head">METRICS EXPLAINED</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    metrics_data = [
        ("🎯", "ACCURACY",  "Overall % of correct predictions out of all predictions made."),
        ("🔬", "PRECISION", "Of all articles predicted as FAKE, how many were actually fake?"),
        ("📡", "RECALL",    "Of ALL actual fake articles, how many did the model catch?"),
        ("⚖️", "F1-SCORE",  "Harmonic mean of Precision and Recall. Best overall single metric."),
    ]
    for col, (icon, title, desc) in zip([m1,m2,m3,m4], metrics_data):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:1.2rem;">
              <div style="font-size:2rem">{icon}</div>
              <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:#00f5ff; letter-spacing:2px; margin:0.5rem 0;">{title}</div>
              <div style="font-size:0.85rem; color:rgba(255,255,255,0.5); line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
# TAB 3: DOCUMENTATION

with tab3:
    st.markdown('<div class="sec-head">SYSTEM DOCUMENTATION</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("# Processing Pipeline")
        steps = [
            ("01", "USER INPUT",      "User pastes news article into the web interface"),
            ("02", "URL REMOVAL",     "All hyperlinks and web addresses removed via regex"),
            ("03", "TEXT CLEANING",   "Special characters, numbers, punctuation removed"),
            ("04", "NORMALIZATION",   "All text converted to lowercase for consistency"),
            ("05", "TOKENIZATION",    "Text split into individual word tokens"),
            ("06", "STOPWORD REMOVAL","Common words (the, is, and) filtered out"),
            ("07", "LEMMATIZATION",   "Words reduced to root form (running → run)"),
            ("08", "TF-IDF",          "Text converted to numerical feature vectors"),
            ("09", "CLASSIFICATION",  "ML model classifies as Real (0) or Fake (1)"),
            ("10", "OUTPUT",          "Result shown with confidence score and visuals"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex; gap:1rem; align-items:flex-start; margin-bottom:0.8rem;">
              <div style="font-family:'Orbitron',monospace; font-size:0.7rem; color:#00f5ff; background:rgba(0,245,255,0.1); border:1px solid rgba(0,245,255,0.2); border-radius:6px; padding:4px 8px; white-space:nowrap; min-width:36px; text-align:center;">{num}</div>
              <div>
                <div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:white; letter-spacing:1px;">{title}</div>
                <div style="font-size:0.85rem; color:rgba(255,255,255,0.5); margin-top:2px;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("###  ML Models")
        st.markdown("""
        **Logistic Regression**
        - Linear classification algorithm
        - Finds mathematical boundary between Real/Fake
        - Uses sigmoid function for probability output
        - Expected accuracy: ~98-99%
        ---
        **Naive Bayes (Multinomial)**
        - Probabilistic Bayesian classifier
        - Calculates P(Fake | words in article)
        - Assumes word independence
        - Expected accuracy: ~93-95%
        ---
        **TF-IDF Vectorizer**
        - max_features: 50,000
        - ngram_range: (1, 2)
        - Captures single words + word pairs
        """)

    st.markdown("---")
    st.markdown("###  Technology Stack")
    st.markdown("""
    | Component | Technology | Purpose |
    |-----------|-----------|---------|
    | Language | Python 3.x | Core programming |
    | Data Handling | Pandas, NumPy | CSV loading & processing |
    | NLP | NLTK | Text cleaning & lemmatization |
    | ML Models | Scikit-learn | Training & prediction |
    | Visualization | Matplotlib, Seaborn, WordCloud | Charts & plots |
    | Web Interface | Streamlit | Frontend + Backend |
    | Dataset | Kaggle Fake News (Bisaillon) | 44,898 labeled articles |
    """)

    st.markdown("---")
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
      <div style="font-family:'Orbitron',monospace; font-size:0.8rem; color:#00f5ff; letter-spacing:3px;">PROJECT DETAILS</div>
      <div style="margin-top:1rem; color:rgba(255,255,255,0.7); font-size:0.95rem; line-height:2;">
        <b>Title:</b> AI-Based Fake News Detection Using ML &amp; NLP<br>
        <b>Degree:</b> BS Computer Science — 8th Semester (FYP)<br>
        <b>SDG Goals:</b> Goal 4 (Quality Education) • Goal 16 (Peace &amp; Justice)
      </div>
    </div>
    """, unsafe_allow_html=True)
