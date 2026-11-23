# 🩺 Aura Health Clinic — Voice AI Receptionist
> **Round 1 SDE / Forward Deployed Engineer (FDE) Assignment — NexovAI**  
> Real-time voice AI receptionist that checks doctor availability, books appointments, and cancels appointments over a phone call interface, paired with a live clinical dashboard.

---

## 🌟 Overview & Key Highlights

This project implements an end-to-end voice AI doctor appointment booking system for a medical clinic. Callers speak naturally in the browser through a phone-style call interface, while clinic staff see a real-time reactive dashboard that updates live whenever an appointment is booked or cancelled.

- **Caller Identification**: Identified **strictly by the phone number spoken aloud** during the call. No login, no passwords, no OTP, and no caller-ID.
- **Strict Verification Rule**: Before confirming any booking or cancellation, the AI agent **reads the phone number back to the caller** to confirm (e.g., *"Just to confirm, I have your number as 9 8 7 6 5 4 3 2 1 0. Is that correct?"*).
- **Executive Frontend Dashboard**: Bespoke clinical dark-mode aesthetic with real-time doctor slot indicators, auto-highlighting appointments board, and live activity streams.
- **Interactive Phone Screen (Bonus)**: Realistic smartphone mockup with call states (*Connecting/Ringing* → *Connected* → *Ended*), live audio waveform visualizer, in-call transcript, booking confirmation ticket, mute controls, and Web Audio synthesized ring/hangup sounds.

---

## 🏗️ Architecture & Pipeline

### Core Voice Flow
```
Caller Mic → 16kHz PCM → Silero VAD → ElevenLabs STT → Google Gemini 2.0 Flash (Function Calling) → ElevenLabs TTS → Browser Audio
                                                                │
                                              ┌─────────────────┴─────────────────┐
                                              ▼                                   ▼
                                     Clinic Tool Handlers               Tap Frame Processor
                                  (Availability / Book / Cancel)      (Live Transcripts & Events)
                                              │                                   │
                                              └─────────────────┬─────────────────┘
                                                                ▼
                                                   FastAPI WebSocket Gateway
                                                                ▼
                                                Reactive Frontend Dashboard
```

### Components
1. **Pipecat (Python)**: Orchestrates the real-time frame pipeline, managing audio buffering, backpressure, turn-taking, and tool scheduling.
2. **Silero VAD**: Low-latency voice activity detection running locally to isolate caller speech and eliminate silence.
3. **ElevenLabs STT (`scribe_v2`)**: Converts caller audio frames to high-fidelity transcribed text.
4. **LLM — Google Gemini (`gemini-2.0-flash`)**: Free-quota, ultra-low latency model executing function calling against the clinic database tools.
5. **ElevenLabs TTS (`eleven_flash_v2_5`)**: Synthesizes warm, natural clinical receptionist speech.
6. **RawPCMSerializer**: Unified WebSocket serializer carrying both 16kHz linear PCM binary audio frames and JSON control messages.
7. **Tap Frame Processor**: Intercepts `TranscriptionFrame`, `TTSTextFrame`, and tool events, broadcasting them in real time to the browser.

---

## 🛠️ Required Tools (Function Calling)

| Tool | Parameters | Purpose |
| :--- | :--- | :--- |
| `check_doctor_availability` | `doctor_name`, `date`, `specialty` | Scans available consultation slots for specialists. |
| `book_appointment` | `patient_phone`, `doctor_name`, `date`, `time_slot`, `patient_name` | Confirms and locks a slot using the caller's spoken phone number. |
| `cancel_appointment` | `patient_phone`, `appointment_id` | Releases an existing booking identified by the spoken phone number. |

---

## 🚀 Step-by-Step Setup Guide

### 1. Clone & Navigate
```bash
git clone <YOUR_REPO_URL>
cd NexovAI
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your API keys:
```env
# Google AI Studio (Free quota: https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# ElevenLabs (https://elevenlabs.io/)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Optional Voice ID (defaults to Rachel)
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```
> **Note**: If API keys are omitted or being configured, the server includes an intelligent simulation fallback so you can test all UI elements, tool states, and audio effects immediately!

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
python server.py
```
Open your browser to:
👉 **`http://localhost:8000`**

---

## 🧪 Testing Order (Verification Protocol)

Use headphones during testing to prevent microphone echo:
1. **Greeting Check**:
   - Click **Try Voice Call**. The phone screen will dial with a dual-tone ring sound and connect.
   - The AI agent greets you: *"Hello! Thank you for calling Aura Health Clinic..."*
2. **Check Availability**:
   - Say: *"Does Dr. Rao have anything tomorrow?"*
   - The agent calls `check_doctor_availability` and offers the available slots (e.g. 10:00 AM, 2:30 PM).
3. **Book an Appointment**:
   - Say: *"I'd like to book tomorrow at 10 AM. My number is 9876543210."*
   - The agent reads back: *"Just to confirm, I have your phone number as 9 8 7 6 5 4 3 2 1 0. Is that correct?"*
   - Say: *"Yes, that's correct."*
   - The agent confirms the booking. Watch the **Live Appointments Board** immediately insert the new row with a glowing green pulse, while the slot chip updates to booked!
4. **Cancel an Appointment**:
   - Say: *"I want to cancel my appointment for phone 9876543210."*
   - The agent confirms the number and cancels the appointment. The board marks it as cancelled and releases the doctor's slot.

---

## 💡 Forward Deployed Engineer (FDE) — Technical Rationale

As an FDE talking directly to enterprise clients, you need clear, honest explanations for architectural choices:

### Q1: *"Why did you use a WebSocket instead of HTTP / REST?"*
> *"Voice AI demands full-duplex, low-latency streaming. In an HTTP request-response model, the browser has to wait for complete sentences or poll constantly. With WebSockets, we establish a single persistent TCP connection where raw 16kHz audio chunks stream upstream while synthesized speech chunks and live transcript events stream downstream concurrently. This eliminates HTTP handshake overhead and reduces perceived response latency from seconds to under 400ms."*

### Q2: *"Why Pipecat for the pipeline?"*
> *"Building real-time voice loops with raw `asyncio` queues becomes fragile when handling interruptions (barge-in), resamplers, and VAD frame boundaries. Pipecat standardizes frame-based streaming across modular processors (transport, VAD, STT, LLM, TTS), with built-in task cancellation when a user interrupts the bot mid-speech."*

### Q3: *"How is user privacy preserved without passwords or OTPs?"*
> *"For rapid clinic phone triage, requiring an OTP introduces excessive friction and abandonment. Instead, the spoken phone number acts as the patient record identifier. The mandatory read-back confirmation step prevents transcription errors before any database mutation occurs."*

---

## 📁 Repository Structure
```
NexovAI/
├── backend/
│   ├── data.py          # Clinic mock database, tool handlers, and change listeners
│   ├── bot.py           # Pipecat pipeline (VAD, ElevenLabs STT/TTS, Gemini LLM, RawPCM, Tap)
│   ├── server.py        # FastAPI server, WebSockets, REST endpoints, static files
│   └── requirements.txt # Python package dependencies
├── frontend/
│   ├── index.html       # Clinic dashboard & smartphone call screen modal
│   ├── style.css        # Executive dark-mode styling, glassmorphism, animations
│   └── app.js           # Web Audio 16kHz PCM capture, playback, waveform canvas, sound synthesis
├── tests/
│   └── test_clinic.py   # Unit test suite verifying all tool actions
├── .env.example         # Template for required environment keys
├── .gitignore           # Git ignore rules (no .env committed)
├── server.py            # Root runner script (`python server.py`)
├── PIPELINE_DESCRIPTION.md # Short, honest pipeline explanation for Google Form submission
└── README.md            # Comprehensive documentation
```

---

## 🎬 Submission Checklist
- [x] Code pushed to GitHub with `.env.example` (no `.env` committed).
- [x] LLM explicitly named: **Google Gemini (`gemini-2.0-flash`)**.
- [x] Short, honest pipeline description in [PIPELINE_DESCRIPTION.md](PIPELINE_DESCRIPTION.md).
- [x] Working end-to-end voice flow (Availability → Read back number → Confirm → Book → Cancel).
- [x] Bespoke, non-generic frontend dashboard with phone screen bonus.
