# Pipeline Description

### Short & Honest Summary (For Submission Form)

I built a real-time voice pipeline using the Pipecat framework in Python, connected to a browser frontend via a bi-directional WebSocket.

Here is how the pipeline operates from end to end:

1. **Audio Ingestion & VAD**: The browser captures microphone audio at 16kHz linear PCM and streams raw binary chunks over the WebSocket. On the server, a custom `RawPCMSerializer` feeds these chunks into Pipecat, where Silero VAD runs locally to detect voice activity and strip background silence.
2. **Speech-to-Text (STT)**: ElevenLabs STT (`scribe_v2`) transcribes caller speech into text frames in real time.
3. **LLM & Function Calling**: Google Gemini (`gemini-2.0-flash`) acts as the reasoning engine. When the caller asks about availability, booking, or cancellation, Gemini triggers one of three Python tool functions (`check_doctor_availability`, `book_appointment`, `cancel_appointment`). 
   - A strict clinical rule is enforced: the agent identifies callers purely by spoken phone number, reads the number back digit by digit, and only calls the tool after explicit confirmation.
4. **Text-to-Speech (TTS) & Playback**: ElevenLabs TTS (`eleven_flash_v2_5`) converts the agent's textual response back into audio streams, which are serialized and sent back over the WebSocket to the browser for low-latency playback.
5. **Dashboard Synchronization (Tap)**: A non-intrusive `Tap` frame processor monitors transcripts and tool events as they flow through the pipeline. Whenever an appointment is booked or cancelled, the backend immediately broadcasts a state update over the WebSocket, causing the frontend appointments board to update with an animated highlight in real time.
