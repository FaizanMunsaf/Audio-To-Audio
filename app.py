import streamlit as st
import whisper
# import ollama
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import time
import threading
from kokoro import KPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceOS · AI Assistant",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS — dark luxury aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080C14;
    color: #E8ECF4;
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0D1F3C 0%, #080C14 50%, #0A0D16 100%);
    min-height: 100vh;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 760px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(139,92,246,0.12));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 100px;
    padding: 4px 16px;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E8ECF4 30%, #38BDF8 70%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -1px;
}
.hero p {
    color: #64748B;
    font-size: 1rem;
    margin-top: 0.75rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* ── Pipeline pills ── */
.pipeline {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0 2rem;
    flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 12px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #94A3B8;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #38BDF8;
    box-shadow: 0 0 8px #38BDF8;
}
.pill.llm .dot { background: #A78BFA; box-shadow: 0 0 8px #A78BFA; }
.pill.tts .dot { background: #34D399; box-shadow: 0 0 8px #34D399; }
.arrow { color: rgba(255,255,255,0.15); font-size: 18px; padding: 0 6px; }

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
}

/* ── Transcript boxes ── */
.transcript-box {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    color: #CBD5E1;
    line-height: 1.6;
    min-height: 60px;
    font-style: italic;
}
.response-box {
    background: rgba(167,139,250,0.05);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 1rem;
    color: #E2E8F0;
    line-height: 1.7;
    min-height: 60px;
}

/* ── Status indicator ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    border-radius: 10px;
    margin: 0.75rem 0;
    font-size: 13px;
    font-weight: 500;
}
.status-idle   { background: rgba(100,116,139,0.1); border: 1px solid rgba(100,116,139,0.2); color: #64748B; }
.status-listen { background: rgba(56,189,248,0.1);  border: 1px solid rgba(56,189,248,0.3);  color: #38BDF8; }
.status-think  { background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.3); color: #A78BFA; }
.status-speak  { background: rgba(52,211,153,0.1);  border: 1px solid rgba(52,211,153,0.3);  color: #34D399; }
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #1E3A5F, #1A2744) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    color: #E8ECF4 !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
div.stButton > button:hover {
    border-color: rgba(56,189,248,0.7) !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.15) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #64748B !important;
}

/* ── Conversation history ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.5rem 0;
}
.msg-ai {
    display: flex;
    justify-content: flex-start;
    margin: 0.5rem 0;
}
.bubble {
    max-width: 80%;
    padding: 10px 16px;
    border-radius: 14px;
    font-size: 14px;
    line-height: 1.5;
}
.bubble-user {
    background: linear-gradient(135deg, #1E3A5F, #162035);
    border: 1px solid rgba(56,189,248,0.2);
    color: #CBD5E1;
    border-bottom-right-radius: 4px;
}
.bubble-ai {
    background: rgba(167,139,250,0.06);
    border: 1px solid rgba(167,139,250,0.15);
    color: #E2E8F0;
    border-bottom-left-radius: 4px;
}

/* ── Sidebar / settings ── */
.model-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    color: #475569;
    margin: 3px 2px;
    font-family: 'Syne', sans-serif;
}

/* Slider & inputs */
div[data-testid="stSlider"] { padding: 0.5rem 0; }
div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8ECF4 !important;
}

/* Audio uploader */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(56,189,248,0.2) !important;
    border-radius: 12px !important;
}

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; }

.footer {
    text-align: center;
    color: #1E293B;
    font-size: 11px;
    padding: 2rem 0 1rem;
    letter-spacing: 1px;
    font-family: 'Syne', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper():
    return whisper.load_model("tiny")

@st.cache_resource(show_spinner=False)
def load_kokoro():
    return KPipeline(lang_code='a')  # 'a' = American English

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "status" not in st.session_state:
    st.session_state.status = "idle"
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def transcribe_audio(file_path: str, model) -> str:
    result = model.transcribe(file_path, language="en", fp16=False)
    return result["text"].strip()

# def ask_ollama(prompt: str, history: list, system_prompt: str) -> str:
#     messages = [{"role": "system", "content": system_prompt}]
#     for turn in history[-6:]:  # last 3 turns context
#         messages.append({"role": "user",      "content": turn["user"]})
#         messages.append({"role": "assistant", "content": turn["ai"]})
#     messages.append({"role": "user", "content": prompt})

#     response = ollama.chat(model="qwen3:4b", messages=messages)
#     return response["message"]["content"].strip()

def ask_gemini(prompt: str, history: list, system_prompt: str) -> str:
    # Build the conversation text
    conversation = system_prompt + "\n\n"
    for turn in history[-6:]:  # last 3 turns context
        conversation += f"User: {turn['user']}\nAssistant: {turn['ai']}\n\n"
    conversation += f"User: {prompt}\nAssistant:"

    url = os.getenv("GEMINI_URL")
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": os.getenv("GEMINI_API_KEY")
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": conversation
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

def synthesize_speech(text: str, pipeline, voice: str = "af_heart") -> np.ndarray:
    samples = []
    generator = pipeline(text, voice=voice, speed=1.0)
    for _, _, audio in generator:
        samples.append(audio)
    if samples:
        return np.concatenate(samples)
    return np.array([])

def play_audio(audio: np.ndarray, sample_rate: int = 24000):
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

def record_audio(duration: int = 5, sample_rate: int = 16000) -> str:
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, recording, sample_rate)
    return tmp.name

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🎙 Voice AI Demo</div>
  <h1>VoiceOS</h1>
  <p>Whisper · Qwen3 · Kokoro — running fully local on your machine</p>
</div>

<div class="pipeline">
  <div class="pill"><span class="dot"></span> Whisper Tiny</div>
  <span class="arrow">→</span>
  <div class="pill llm"><span class="dot"></span> Qwen3 4B</div>
  <span class="arrow">→</span>
  <div class="pill tts"><span class="dot"></span> Kokoro TTS</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, concise voice assistant. Keep responses under 3 sentences unless asked for more detail.",
        height=120
    )
    rec_duration = st.slider("Recording duration (sec)", 3, 15, 5)
    tts_voice = st.selectbox("Kokoro Voice", ["af_heart", "af_bella", "am_adam", "bf_emma"])
    auto_play = st.toggle("Auto-play response", value=True)
    st.markdown("---")
    st.markdown("**Models active:**")
    st.markdown('<span class="model-chip">🔵 whisper-tiny</span>', unsafe_allow_html=True)
    st.markdown('<span class="model-chip">🟣 qwen3:4b (ollama)</span>', unsafe_allow_html=True)
    st.markdown('<span class="model-chip">🟢 kokoro-tts</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────
status_map = {
    "idle":    ("status-idle",   "●",  "Ready — choose an input method below"),
    "listen":  ("status-listen", "◉",  "Listening…"),
    "transcribe": ("status-listen", "◉", "Transcribing audio…"),
    "think":   ("status-think",  "◉",  "Qwen3 is thinking…"),
    "speak":   ("status-speak",  "◉",  "Speaking response…"),
}
sc, icon, label = status_map.get(st.session_state.status, status_map["idle"])
st.markdown(f"""
<div class="status-bar {sc}">
  <span class="status-dot"></span>
  <span>{label}</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Input</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎤  Record Voice", "📁  Upload Audio"])

user_audio_path = None

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⏺ Record Now", key="rec"):
            with st.spinner(f"Recording for {rec_duration}s…"):
                st.session_state.status = "listen"
                user_audio_path = record_audio(duration=rec_duration)
            st.success("Recording saved!")

with tab2:
    uploaded = st.file_uploader("Drop a .wav or .mp3 file", type=["wav", "mp3", "m4a"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False)
        tmp.write(uploaded.read())
        tmp.flush()
        user_audio_path = tmp.name
        st.audio(uploaded)

st.markdown("</div>", unsafe_allow_html=True)

# Text fallback
st.markdown('<div class="card"><div class="card-title">Or type a message</div>', unsafe_allow_html=True)
text_input = st.text_input("", placeholder="Type your question here…", label_visibility="collapsed")
send_text = st.button("Send Text →", key="send_text")
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────
user_text = ""

# From audio
if user_audio_path:
    st.session_state.status = "transcribe"
    with st.spinner("Transcribing with Whisper Tiny…"):
        whisper_model = load_whisper()
        user_text = transcribe_audio(user_audio_path, whisper_model)
    st.session_state.last_transcript = user_text

# From text input
if send_text and text_input.strip():
    user_text = text_input.strip()
    st.session_state.last_transcript = user_text

# Run LLM + TTS
if user_text:
    # Show transcript
    st.markdown(f"""
    <div class="card">
      <div class="card-title">You said</div>
      <div class="transcript-box">"{user_text}"</div>
    </div>
    """, unsafe_allow_html=True)

    # LLM
    st.session_state.status = "think"
    with st.spinner("Gemini Flash is thinking…"):
        ai_reply = ask_gemini(user_text, st.session_state.history, system_prompt)

    st.session_state.last_response = ai_reply
    st.session_state.history.append({"user": user_text, "ai": ai_reply})

    # Show response
    st.markdown(f"""
    <div class="card">
      <div class="card-title">AI Response</div>
      <div class="response-box">{ai_reply}</div>
    </div>
    """, unsafe_allow_html=True)

    # TTS
    if auto_play:
        st.session_state.status = "speak"
        with st.spinner("Synthesizing with Kokoro TTS…"):
            kokoro_pipeline = load_kokoro()
            audio_arr = synthesize_speech(ai_reply, kokoro_pipeline, voice=tts_voice)
            if audio_arr.size > 0:
                # Save to tmp and offer download + playback
                tts_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tts_tmp.name, audio_arr, 24000)
                st.audio(tts_tmp.name, format="audio/wav")
                if auto_play:
                    play_audio(audio_arr)

    st.session_state.status = "idle"

# ─────────────────────────────────────────────
# CONVERSATION HISTORY
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="card"><div class="card-title">Conversation History</div>', unsafe_allow_html=True)
    for turn in st.session_state.history[-5:]:
        st.markdown(f'<div class="msg-user"><div class="bubble bubble-user">🎙 {turn["user"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-ai"><div class="bubble bubble-ai">🤖 {turn["ai"]}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear History", key="clear"):
            st.session_state.history = []
            st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="footer">VOICEOS · LOCAL AI · WHISPER + QWEN3 + KOKORO</div>', unsafe_allow_html=True)