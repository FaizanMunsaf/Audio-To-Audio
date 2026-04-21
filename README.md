# 🎙️ VoiceOS — Setup Guide

> Whisper Tiny → Qwen3:4b (Ollama) → Kokoro TTS → Streamlit UI

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Python | 3.10+ | `python --version` |
| Ollama | installed | `ollama --version` |
| Qwen3:4b weights | downloaded | `ollama list` |
| ffmpeg | any | `ffmpeg -version` |
| PortAudio | any | (for sounddevice) |

---

## Step 1 — Install ffmpeg & PortAudio

### macOS
```bash
brew install ffmpeg portaudio
```

### Ubuntu / Debian
```bash
sudo apt update && sudo apt install -y ffmpeg portaudio19-dev
```

### Windows
```bash
# Install ffmpeg from https://ffmpeg.org/download.html
# Install portaudio via: pip install pipwin && pipwin install pyaudio
```

---

## Step 2 — Create Python virtual environment

```bash
# Navigate to the project folder
cd voice_assistant

# Create venv
python -m venv venv

# Activate
source venv/bin/activate        # macOS/Linux
# or
venv\Scripts\activate           # Windows
```

---

## Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Whisper will auto-download the `tiny` model (~75MB) on first run.

---

## Step 4 — Verify Ollama + Qwen3:4b

```bash
# Make sure Ollama is running
ollama serve &

# Confirm qwen3:4b is available
ollama list
# Should show: qwen3:4b

# Quick test
ollama run qwen3:4b "Say hello in one sentence"
```

---

## Step 5 — Install Kokoro TTS

Kokoro is a lightweight open-source TTS engine:

```bash
pip install kokoro>=0.9.2

# Kokoro will auto-download voice weights on first use (~200MB)
# Voices available: af_heart, af_bella, am_adam, bf_emma
```

---

## Step 6 — Run the app

```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501** 🎉

---

## How It Works

```
[Your Voice]
     ↓
[Whisper Tiny]  →  transcribes speech to text (local, fast ~1-2s)
     ↓
[Qwen3:4b via Ollama]  →  generates intelligent response (local LLM)
     ↓
[Kokoro TTS]  →  converts response text to natural speech (local)
     ↓
[Streamlit UI]  →  plays audio + shows conversation history
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named whisper` | `pip install openai-whisper` |
| `ollama.ResponseError` | Make sure `ollama serve` is running |
| `No module named kokoro` | `pip install kokoro` |
| No microphone / sounddevice error | Install PortAudio (Step 1) |
| Slow transcription | Normal on CPU; Whisper tiny is ~1-3s |
| Ollama connection refused | Run `ollama serve` in a separate terminal |

---

## Project Structure

```
voice_assistant/
├── app.py              ← Main Streamlit app
├── requirements.txt    ← Python dependencies
└── SETUP.md            ← This file
```

---

## Tips for Client Demo

- Use **"Record Voice"** tab for live demo
- Set recording to **5 seconds** — enough for a clear question
- Try questions like:
  - *"What's the capital of France?"*
  - *"Explain quantum computing in simple terms"*
  - *"Write me a short poem about technology"*
- The **sidebar** lets you swap voices and adjust the system prompt live
- All processing is **100% local** — no internet needed after setup

---

*Built with ❤️ — Whisper + Ollama + Kokoro + Streamlit*