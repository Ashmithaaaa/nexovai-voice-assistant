/**
 * app.js - Aura Health Clinic Frontend & Voice Agent Controller
 *
 * Implements:
 *   - Real-time WebSocket connection to backend voice pipeline
 *   - Web Audio 16kHz linear PCM microphone recording & streaming
 *   - Seamless PCM audio playback & Web Speech fallback
 *   - Responsive audio waveform oscilloscope visualizer
 *   - Synthesized phone ringtone & disconnect sound effects (Web Audio API)
 *   - Live reactive doctors directory & auto-highlighting appointments board
 */

// Application State
const AppState = {
  ws: null,
  wsConnected: false,
  callActive: false,
  isMuted: false,
  callDurationSec: 0,
  callTimerInterval: null,
  audioCtx: null,
  micStream: null,
  audioInputNode: null,
  scriptProcessor: null,
  analyser: null,
  recognition: null,
  speechSynth: window.speechSynthesis,
  doctors: [],
  appointments: [],
  selectedFilter: 'all',
  lastBookedAptId: null,
  lastCancelledAptId: null,
};

// DOM Element References
const DOM = {
  btnOpenCall: document.getElementById('btnOpenCall'),
  btnResetData: document.getElementById('btnResetData'),
  phoneModalBackdrop: document.getElementById('phoneModalBackdrop'),
  phoneDevice: document.getElementById('phoneDevice'),
  callStateBadge: document.getElementById('callStateBadge'),
  callStateText: document.getElementById('callStateText'),
  callTimer: document.getElementById('callTimer'),
  phoneTranscript: document.getElementById('phoneTranscript'),
  phoneTicketCard: document.getElementById('phoneTicketCard'),
  ticketTitle: document.getElementById('ticketTitle'),
  ticketDoctor: document.getElementById('ticketDoctor'),
  ticketSlot: document.getElementById('ticketSlot'),
  ticketPhone: document.getElementById('ticketPhone'),
  waveformCanvas: document.getElementById('waveformCanvas'),
  btnToggleMute: document.getElementById('btnToggleMute'),
  muteIcon: document.getElementById('muteIcon'),
  muteText: document.getElementById('muteText'),
  btnEndCall: document.getElementById('btnEndCall'),
  btnMicCheck: document.getElementById('btnMicCheck'),
  phoneHomeBar: document.getElementById('phoneHomeBar'),
  phoneClock: document.getElementById('phoneClock'),
  doctorsListContainer: document.getElementById('doctorsListContainer'),
  appointmentsTableBody: document.getElementById('appointmentsTableBody'),
  activityFeed: document.getElementById('activityFeed'),
  logCounter: document.getElementById('logCounter'),
  specialtyFilter: document.getElementById('specialtyFilter'),
  statDoctorsCount: document.getElementById('statDoctorsCount'),
  statAvailableSlots: document.getElementById('statAvailableSlots'),
  statAppointmentsCount: document.getElementById('statAppointmentsCount'),
  toastContainer: document.getElementById('toastContainer'),
  phoneTextInput: document.getElementById('phoneTextInput'),
  btnSendPhoneText: document.getElementById('btnSendPhoneText'),
};

// ---------------------------------------------------------------------------
// 1. Initial Setup & Event Listeners
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  updateClock();
  setInterval(updateClock, 10000);

  // Load clinic data immediately via REST
  loadInitialData();

  initWebSocket();
  setupFilterPills();
  setupQuickPrompts();
  setupTextInputBar();
  initWaveformVisualizer();

  DOM.btnOpenCall.addEventListener('click', startPhoneCall);
  DOM.btnEndCall.addEventListener('click', endPhoneCall);
  DOM.btnToggleMute.addEventListener('click', toggleMute);
  DOM.btnMicCheck.addEventListener('click', testMicrophone);
  DOM.phoneHomeBar.addEventListener('click', endPhoneCall);
  DOM.btnResetData.addEventListener('click', resetDemoData);
});

async function loadInitialData() {
  try {
    const [docsRes, aptsRes] = await Promise.all([
      fetch('/api/doctors'),
      fetch('/api/appointments')
    ]);
    const docsData = await docsRes.json();
    const aptsData = await aptsRes.json();
    if (docsData && docsData.doctors) {
      AppState.doctors = docsData.doctors;
      renderDoctors();
    }
    if (aptsData && aptsData.appointments) {
      AppState.appointments = aptsData.appointments;
      renderAppointments();
    }
    updateStatsCounters();
  } catch (e) {
    console.warn('[Data] Initial REST load notice:', e);
  }
}

function setupTextInputBar() {
  function handleSend() {
    if (!DOM.phoneTextInput) return;
    const val = DOM.phoneTextInput.value.trim();
    if (!val) return;

    if (!AppState.callActive) {
      showToast('Click "Try Voice Call" first to start the call session.', 'info');
      return;
    }

    addTranscriptMessage('user', val, true);
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
      AppState.ws.send(JSON.stringify({ type: 'user_transcript', text: val }));
    }
    DOM.phoneTextInput.value = '';
  }

  if (DOM.btnSendPhoneText) {
    DOM.btnSendPhoneText.addEventListener('click', handleSend);
  }
  if (DOM.phoneTextInput) {
    DOM.phoneTextInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSend();
      }
    });
  }
}

function updateClock() {
  const now = new Date();
  const hrs = String(now.getHours()).padStart(2, '0');
  const mins = String(now.getMinutes()).padStart(2, '0');
  if (DOM.phoneClock) DOM.phoneClock.textContent = `${hrs}:${mins}`;
}

// ---------------------------------------------------------------------------
// 2. WebSocket Connection & Event Handler
// ---------------------------------------------------------------------------
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  console.log(`[WebSocket] Connecting to ${wsUrl}...`);
  AppState.ws = new WebSocket(wsUrl);
  AppState.ws.binaryType = 'arraybuffer';

  AppState.ws.onopen = () => {
    console.log('[WebSocket] Connected successfully.');
    AppState.wsConnected = true;
    updateServerBadge(true);
    addActivityLog('WS_CONNECT', 'Established real-time channel with Voice AI backend.');
  };

  AppState.ws.onmessage = async (event) => {
    // 1. Check if message is binary PCM audio from backend TTS
    if (event.data instanceof ArrayBuffer) {
      playRawAudioChunk(event.data);
      return;
    }

    // 2. Parse text message
    try {
      const data = JSON.parse(event.data);
      handleServerEvent(data);
    } catch (err) {
      console.warn('[WebSocket] Non-JSON message received:', event.data);
    }
  };

  AppState.ws.onclose = () => {
    console.warn('[WebSocket] Connection closed. Retrying in 2s...');
    AppState.wsConnected = false;
    updateServerBadge(false);
    setTimeout(initWebSocket, 2000);
  };

  AppState.ws.onerror = (err) => {
    console.error('[WebSocket] Error:', err);
    updateServerBadge(false);
  };
}

function updateServerBadge(online) {
  const badge = document.getElementById('serverStatusBadge');
  const pulse = document.getElementById('serverPulse');
  const text = document.getElementById('serverStatusText');
  if (!text || !pulse) return;

  if (online) {
    pulse.style.backgroundColor = 'var(--accent-emerald)';
    pulse.style.boxShadow = '0 0 10px var(--accent-emerald)';
    text.textContent = 'Server Online • Ready';
    if (badge) badge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
  } else {
    pulse.style.backgroundColor = 'var(--accent-rose)';
    pulse.style.boxShadow = '0 0 10px var(--accent-rose)';
    text.textContent = 'Server Offline (Run python server.py)';
    if (badge) badge.style.borderColor = 'rgba(244, 63, 94, 0.4)';
  }
}

function handleServerEvent(data) {
  const type = data.type;

  // 1. Clinic State Synchronization
  if (type === 'clinic_update') {
    if (data.doctors) {
      AppState.doctors = data.doctors;
      renderDoctors();
    }
    if (data.appointments) {
      AppState.appointments = data.appointments;
      renderAppointments();
    }
    updateStatsCounters();
  }

  // 2. Live Conversation Transcripts
  else if (type === 'transcript') {
    addTranscriptMessage(data.role, data.text, data.is_final);

    // Speak agent text locally if audio wasn't streamed over PCM
    if (data.role === 'agent') {
      speakAgentFallback(data.text);
    }
  }

  // 3. Agent State Transitions
  else if (type === 'state') {
    updateCallStatus(data.status);
  }

  // 4. Tool Execution Events
  else if (type === 'tool_event') {
    handleToolEvent(data);
  }
}

function handleToolEvent(event) {
  const action = event.action;
  const details = event.details || {};

  if (action === 'check_doctor_availability') {
    addActivityLog('TOOL: AVAILABILITY', `Scanned doctor slots: ${details.status || 'success'}`);
    showToast(`Checked doctor availability (${details.count || 0} specialists)`, 'info');
  }

  else if (action === 'book_appointment') {
    addActivityLog('TOOL: BOOKING', `Confirmed for ${details.doctor} (${details.time_slot})`);
    showToast(`Booked appointment with ${details.doctor}!`, 'success');
    showPhoneTicket('Appointment Confirmed', details.doctor, `${details.date} at ${details.time_slot}`, details.patient_phone);
    AppState.lastBookedAptId = details.appointment_id;
  }

  else if (action === 'cancel_appointment') {
    addActivityLog('TOOL: CANCEL', `Appointment cancelled for phone ${details.patient_phone}`);
    showToast(`Appointment with ${details.doctor} cancelled`, 'info');
    showPhoneTicket('Appointment Cancelled', details.doctor, `${details.date} at ${details.time_slot}`, details.patient_phone);
    AppState.lastCancelledAptId = details.appointment_id;
  }
}

// ---------------------------------------------------------------------------
// 3. Phone Call Screen Management (Bonus Feature)
// ---------------------------------------------------------------------------
async function startPhoneCall() {
  if (AppState.callActive) return;

  if (!AppState.wsConnected) {
    showToast('Cannot connect: Server is offline. Please run "python server.py" in VS Code terminal.', 'info');
    return;
  }

  // Show Phone Modal
  DOM.phoneModalBackdrop.classList.add('active');
  updateCallStatus('ringing');
  DOM.callTimer.textContent = '00:00';
  AppState.callDurationSec = 0;

  // Play synthesized dual-tone telephone ringing sound
  playTelephoneRingSound();

  // Simulate call connection progression
  setTimeout(async () => {
    playCallConnectBeep();
    AppState.callActive = true;
    updateCallStatus('connected');

    // Start timer
    AppState.callTimerInterval = setInterval(() => {
      AppState.callDurationSec++;
      const mins = String(Math.floor(AppState.callDurationSec / 60)).padStart(2, '0');
      const secs = String(AppState.callDurationSec % 60).padStart(2, '0');
      DOM.callTimer.textContent = `${mins}:${secs}`;
    }, 1000);

    // Initialize Microphone & Audio streaming
    await startMicrophoneCapture();
    initSpeechRecognitionFallback();

    // Notify backend
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
      AppState.ws.send(JSON.stringify({ type: 'call_started' }));
    }
  }, 1800);
}

function endPhoneCall() {
  if (!AppState.callActive && !DOM.phoneModalBackdrop.classList.contains('active')) return;

  playDisconnectTone();
  AppState.callActive = false;

  if (AppState.callTimerInterval) {
    clearInterval(AppState.callTimerInterval);
    AppState.callTimerInterval = null;
  }

  stopMicrophoneCapture();
  if (AppState.recognition) {
    try { AppState.recognition.stop(); } catch (e) {}
  }
  if (AppState.speechSynth) {
    AppState.speechSynth.cancel();
  }

  updateCallStatus('ended');

  if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
    AppState.ws.send(JSON.stringify({ type: 'call_ended' }));
  }

  setTimeout(() => {
    DOM.phoneModalBackdrop.classList.remove('active');
    DOM.phoneTicketCard.style.display = 'none';
  }, 1200);
}

function toggleMute() {
  AppState.isMuted = !AppState.isMuted;
  DOM.btnToggleMute.classList.toggle('muted', AppState.isMuted);
  DOM.muteText.textContent = AppState.isMuted ? 'Unmute' : 'Mute';
  showToast(AppState.isMuted ? 'Microphone muted' : 'Microphone unmuted', 'info');
}

function updateCallStatus(status) {
  const map = {
    'ringing': { text: 'Calling Aura Clinic...', dot: '#F59E0B' },
    'connected': { text: 'Call Connected', dot: '#10B981' },
    'listening': { text: 'Listening to You...', dot: '#0EA5E9' },
    'thinking': { text: 'Checking Schedule...', dot: '#8B5CF6' },
    'speaking': { text: 'Aura Speaking...', dot: '#10B981' },
    'idle': { text: 'Connected', dot: '#10B981' },
    'ended': { text: 'Call Ended', dot: '#F43F5E' },
  };

  const current = map[status] || { text: status, dot: '#0EA5E9' };
  DOM.callStateText.textContent = current.text;
  const dot = DOM.callStateBadge.querySelector('.state-dot');
  if (dot) dot.style.backgroundColor = current.dot;
}

// ---------------------------------------------------------------------------
// 4. Microphone Audio Streaming (16kHz PCM) & Web Audio
// ---------------------------------------------------------------------------
async function startMicrophoneCapture() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    AppState.audioCtx = new AudioContextClass({ sampleRate: 16000 });

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    AppState.micStream = stream;
    AppState.audioInputNode = AppState.audioCtx.createMediaStreamSource(stream);

    // Setup Analyser for Canvas Waveform
    AppState.analyser = AppState.audioCtx.createAnalyser();
    AppState.analyser.fftSize = 256;
    AppState.audioInputNode.connect(AppState.analyser);

    // ScriptProcessor to capture raw PCM Float32 and downsample/convert to Int16
    AppState.scriptProcessor = AppState.audioCtx.createScriptProcessor(2048, 1, 1);
    AppState.scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
      if (!AppState.callActive || AppState.isMuted) return;

      const inputBuffer = audioProcessingEvent.inputBuffer;
      const inputData = inputBuffer.getChannelData(0);

      // Convert Float32 to 16-bit linear PCM
      const pcm16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        let s = Math.max(-1, Math.min(1, inputData[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Stream binary PCM chunks over WebSocket to Pipecat backend
      if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
        AppState.ws.send(pcm16.buffer);
      }
    };

    AppState.audioInputNode.connect(AppState.scriptProcessor);
    AppState.scriptProcessor.connect(AppState.audioCtx.destination);
    console.log('[Audio] 16kHz PCM Microphone stream connected.');
  } catch (err) {
    console.error('[Audio] Microphone permission denied or unavailable:', err);
    showToast('Microphone access unavailable. Using browser speech recognition.', 'info');
  }
}

function stopMicrophoneCapture() {
  if (AppState.scriptProcessor) {
    AppState.scriptProcessor.disconnect();
    AppState.scriptProcessor = null;
  }
  if (AppState.audioInputNode) {
    AppState.audioInputNode.disconnect();
    AppState.audioInputNode = null;
  }
  if (AppState.micStream) {
    AppState.micStream.getTracks().forEach((track) => track.stop());
    AppState.micStream = null;
  }
  if (AppState.audioCtx && AppState.audioCtx.state !== 'closed') {
    AppState.audioCtx.close();
    AppState.audioCtx = null;
  }
}

function testMicrophone() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('WebRTC Audio not supported in this browser.', 'info');
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then((stream) => {
      showToast('Microphone working properly!', 'success');
      stream.getTracks().forEach((t) => t.stop());
    })
    .catch((err) => {
      showToast(`Microphone error: ${err.message}`, 'info');
    });
}

// ---------------------------------------------------------------------------
// 5. Speech Recognition & Synthesis Fallbacks
// ---------------------------------------------------------------------------
function initSpeechRecognitionFallback() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) return;

  AppState.recognition = new SpeechRecognition();
  AppState.recognition.continuous = true;
  AppState.recognition.interimResults = true;
  AppState.recognition.lang = 'en-US';

  AppState.recognition.onresult = (event) => {
    let finalTranscript = '';
    let interimTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    if (finalTranscript.trim()) {
      addTranscriptMessage('user', finalTranscript.trim(), true);
      // Forward user text to WebSocket backend
      if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
        AppState.ws.send(JSON.stringify({
          type: 'user_transcript',
          text: finalTranscript.trim(),
        }));
      }
    } else if (interimTranscript.trim()) {
      addTranscriptMessage('user', interimTranscript.trim(), false);
    }
  };

  AppState.recognition.onerror = (e) => {
    console.log('[SpeechRecognition] Event:', e.error);
  };

  try {
    AppState.recognition.start();
  } catch (err) {}
}

function speakAgentFallback(text) {
  if (!AppState.speechSynth) return;
  if (AppState.audioCtx && AppState.audioCtx.state === 'running' && AppState.isPlayingRemoteAudio) {
    return;
  }
  AppState.speechSynth.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  // Select clear, pleasant voice if available
  const voices = AppState.speechSynth.getVoices();
  const preferred = voices.find(v => v.lang.startsWith('en') && (
    v.name.includes('Natural') || v.name.includes('Jenny') || v.name.includes('Samantha') || 
    v.name.includes('Google US English') || v.name.includes('Zira') || v.name.includes('Female')
  ));
  if (preferred) utterance.voice = preferred;

  utterance.onstart = () => {
    updateCallStatus('speaking');
    AppState.isAgentSpeaking = true;
  };
  utterance.onend = () => {
    updateCallStatus('listening');
    AppState.isAgentSpeaking = false;
  };
  utterance.onerror = () => {
    AppState.isAgentSpeaking = false;
  };

  AppState.speechSynth.speak(utterance);
}

function playRawAudioChunk(arrayBuffer) {
  AppState.isPlayingRemoteAudio = true;
  // Create or reuse audio context for remote PCM playback
  if (!AppState.playbackCtx) {
    AppState.playbackCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  }

  const int16Array = new Int16Array(arrayBuffer);
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0;
  }

  const audioBuffer = AppState.playbackCtx.createBuffer(1, float32Array.length, 16000);
  audioBuffer.getChannelData(0).set(float32Array);

  const source = AppState.playbackCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(AppState.playbackCtx.destination);
  source.start();

  source.onended = () => {
    AppState.isPlayingRemoteAudio = false;
  };
}

// ---------------------------------------------------------------------------
// 6. Sound Effects Synthesizer (Telephone Dual-Tone & Chimes)
// ---------------------------------------------------------------------------
function playTelephoneRingSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gainNode = ctx.createGain();

    // US standard ringback tone frequencies: 440 Hz + 480 Hz
    osc1.frequency.value = 440;
    osc2.frequency.value = 480;

    // Pulse volume
    gainNode.gain.setValueAtTime(0, ctx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.1);
    gainNode.gain.setValueAtTime(0.12, ctx.currentTime + 1.2);
    gainNode.gain.linearRampToValueAtTime(0, ctx.currentTime + 1.4);

    osc1.connect(gainNode);
    osc2.connect(gainNode);
    gainNode.connect(ctx.destination);

    osc1.start();
    osc2.start();
    osc1.stop(ctx.currentTime + 1.5);
    osc2.stop(ctx.currentTime + 1.5);
  } catch (e) {}
}

function playCallConnectBeep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.frequency.setValueAtTime(600, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(900, ctx.currentTime + 0.15);

    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.2);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.2);
  } catch (e) {}
}

function playDisconnectTone() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.frequency.setValueAtTime(480, ctx.currentTime);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.35);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.35);
  } catch (e) {}
}

// ---------------------------------------------------------------------------
// 7. Audio Waveform Visualizer
// ---------------------------------------------------------------------------
function initWaveformVisualizer() {
  const canvas = DOM.waveformCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function draw() {
    requestAnimationFrame(draw);
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (!AppState.callActive) {
      // Idle line
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 2;
      ctx.stroke();
      return;
    }

    let dataArray = new Uint8Array(64);
    if (AppState.isAgentSpeaking) {
      // Dynamic pulsating waveform while receptionist is speaking
      for (let i = 0; i < 64; i++) {
        dataArray[i] = Math.abs(Math.sin(Date.now() * 0.008 + i * 0.25)) * 170 + 40;
      }
    } else if (AppState.analyser) {
      AppState.analyser.getByteFrequencyData(dataArray);
    } else {
      // Gentle synthetic pulse if analyser is preparing
      for (let i = 0; i < 64; i++) {
        dataArray[i] = Math.sin(Date.now() * 0.005 + i) * 30 + 35;
      }
    }

    const barWidth = (width / 32);
    let x = 0;

    for (let i = 0; i < 32; i++) {
      const barHeight = Math.max(4, (dataArray[i] / 255) * height * 0.9);

      // Cyan to Emerald gradient
      const grad = ctx.createLinearGradient(0, height / 2 - barHeight / 2, 0, height / 2 + barHeight / 2);
      grad.addColorStop(0, '#0EA5E9');
      grad.addColorStop(1, '#10B981');

      ctx.fillStyle = grad;
      ctx.fillRect(x, (height - barHeight) / 2, barWidth - 3, barHeight);

      x += barWidth;
    }
  }

  draw();
}

// ---------------------------------------------------------------------------
// 8. Transcript Management & Live UI
// ---------------------------------------------------------------------------
function addTranscriptMessage(role, text, isFinal = true) {
  if (!text || !text.trim()) return;

  const viewport = DOM.phoneTranscript;
  if (!viewport) return;

  // Check if we can update existing interim user bubble
  if (role === 'user' && !isFinal) {
    let interim = viewport.querySelector('.interim-bubble');
    if (!interim) {
      interim = document.createElement('div');
      interim.className = 'transcript-msg user-msg interim-bubble';
      interim.innerHTML = `
        <div class="msg-sender">You (Speaking...)</div>
        <div class="msg-bubble">${escapeHtml(text)}</div>
      `;
      viewport.appendChild(interim);
    } else {
      interim.querySelector('.msg-bubble').textContent = text;
    }
    viewport.scrollTop = viewport.scrollHeight;
    return;
  }

  // Remove any lingering interim bubble
  const interim = viewport.querySelector('.interim-bubble');
  if (interim) interim.remove();

  const msgDiv = document.createElement('div');
  msgDiv.className = `transcript-msg ${role === 'user' ? 'user-msg' : 'agent-msg'}`;

  const senderName = role === 'user' ? 'Caller' : 'Aura Receptionist';
  msgDiv.innerHTML = `
    <div class="msg-sender">${senderName}</div>
    <div class="msg-bubble">${escapeHtml(text)}</div>
  `;

  viewport.appendChild(msgDiv);
  viewport.scrollTop = viewport.scrollHeight;

  // Also add brief entry to dashboard activity feed
  addActivityLog(role === 'user' ? 'USER_SPEECH' : 'AGENT_SPEECH', text.slice(0, 50) + (text.length > 50 ? '...' : ''));
}

function showPhoneTicket(title, doctor, slot, phone) {
  DOM.ticketTitle.textContent = title;
  DOM.ticketDoctor.textContent = doctor;
  DOM.ticketSlot.textContent = slot;
  DOM.ticketPhone.textContent = phone;
  DOM.phoneTicketCard.style.display = 'block';

  // Auto-scroll inside phone
  DOM.phoneTranscript.scrollTop = DOM.phoneTranscript.scrollHeight;
}

// ---------------------------------------------------------------------------
// 9. Doctors & Slots Rendering
// ---------------------------------------------------------------------------
function setupFilterPills() {
  const pills = DOM.specialtyFilter.querySelectorAll('.filter-pill');
  pills.forEach((pill) => {
    pill.addEventListener('click', () => {
      pills.forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
      AppState.selectedFilter = pill.getAttribute('data-filter');
      renderDoctors();
    });
  });
}

function renderDoctors() {
  const container = DOM.doctorsListContainer;
  if (!container) return;

  const filter = AppState.selectedFilter;
  const filteredDocs = filter === 'all'
    ? AppState.doctors
    : AppState.doctors.filter((d) => d.specialty.toLowerCase() === filter.toLowerCase());

  if (filteredDocs.length === 0) {
    container.innerHTML = `<div class="loading-placeholder">No doctors found for ${filter}.</div>`;
    return;
  }

  container.innerHTML = filteredDocs.map((doc) => {
    const slotsHtml = doc.available_slots.map((slot) => {
      const isAvailable = slot.status === 'available';
      const slotLabel = `${slot.date} ${slot.time}`;
      return `
        <span class="slot-chip ${isAvailable ? 'available' : 'booked'}"
              title="${isAvailable ? 'Click to preview spoken prompt' : 'Already Booked'}"
              onclick="onSlotClicked('${doc.name}', '${slot.date}', '${slot.time}', ${isAvailable})">
          ${slotLabel}
        </span>
      `;
    }).join('');

    return `
      <div class="doctor-item-card" data-doc-id="${doc.id}">
        <div class="doc-avatar-box">
          <img class="doc-avatar-img" src="${doc.avatar}" alt="${doc.name}" />
          <span class="doc-rating-badge">&#9733; ${doc.rating}</span>
        </div>
        <div class="doc-info-col">
          <div class="doc-header-row">
            <h3 class="doc-name">${doc.name}</h3>
            <span class="doc-specialty-tag">${doc.specialty}</span>
          </div>
          <div class="slots-chips-title">Available Slots:</div>
          <div class="slots-chips-grid">
            ${slotsHtml}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

window.onSlotClicked = function(docName, date, time, isAvailable) {
  if (!isAvailable) {
    showToast('This slot is already booked.', 'info');
    return;
  }
  const prompt = `I'd like to book ${docName} for ${date} at ${time}. My phone number is 9876543210.`;
  showToast(`Speech tip: Say "${prompt}" during your call!`, 'info');

  if (AppState.callActive) {
    // Fill prompt chip or send directly
    addTranscriptMessage('user', prompt, true);
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
      AppState.ws.send(JSON.stringify({ type: 'user_transcript', text: prompt }));
    }
  }
};

// ---------------------------------------------------------------------------
// 10. Appointments Board Rendering with Real-Time Highlight Glow
// ---------------------------------------------------------------------------
function renderAppointments() {
  const tbody = DOM.appointmentsTableBody;
  if (!tbody) return;

  if (AppState.appointments.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted); padding: 2rem;">No appointments booked yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = AppState.appointments.map((apt) => {
    const isConfirmed = apt.status === 'Confirmed';
    const isNewBooking = apt.id === AppState.lastBookedAptId;
    const isNewCancellation = apt.id === AppState.lastCancelledAptId;

    let rowClass = '';
    if (isNewBooking) rowClass = 'row-newly-booked';
    else if (isNewCancellation) rowClass = 'row-newly-cancelled';

    return `
      <tr class="${rowClass}" data-apt-id="${apt.id}">
        <td>
          <span class="status-badge ${isConfirmed ? 'confirmed' : 'cancelled'}">
            ${isConfirmed ? '&#9679; Confirmed' : '&#9679; Cancelled'}
          </span>
        </td>
        <td>
          <strong style="color: #F8FAFC;">${apt.doctor_name}</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${apt.doctor_specialty || ''}</div>
        </td>
        <td>
          <span class="phone-digits">${formatPhone(apt.patient_phone)}</span>
        </td>
        <td>
          <span style="color: #BAE6FD; font-weight: 600;">${apt.date}</span> at ${apt.time_slot}
        </td>
        <td>
          ${isConfirmed ? `
            <button class="btn-inline-cancel" onclick="cancelAppointmentInline('${apt.patient_phone}', '${apt.id}')" title="Cancel this appointment">
              Cancel
            </button>
          ` : `<span style="font-size: 0.75rem; color: var(--text-muted);">Released</span>`}
        </td>
      </tr>
    `;
  }).join('');
}

window.cancelAppointmentInline = function(phone, aptId) {
  if (confirm(`Do you want to cancel the appointment for phone ${phone}?`)) {
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
      AppState.ws.send(JSON.stringify({
        type: 'user_transcript',
        text: `Please cancel my appointment for phone number ${phone}.`,
      }));
    }
  }
};

function updateStatsCounters() {
  if (DOM.statDoctorsCount) DOM.statDoctorsCount.textContent = AppState.doctors.length;

  let totalSlots = 0;
  AppState.doctors.forEach((doc) => {
    totalSlots += doc.available_slots.filter((s) => s.status === 'available').length;
  });
  if (DOM.statAvailableSlots) DOM.statAvailableSlots.textContent = totalSlots;

  const confirmedCount = AppState.appointments.filter((a) => a.status === 'Confirmed').length;
  if (DOM.statAppointmentsCount) DOM.statAppointmentsCount.textContent = confirmedCount;
}

// ---------------------------------------------------------------------------
// 11. Quick Prompts & Helpers
// ---------------------------------------------------------------------------
function setupQuickPrompts() {
  const chips = document.querySelectorAll('.prompt-chip');
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const text = chip.getAttribute('data-text');
      if (text && AppState.callActive) {
        addTranscriptMessage('user', text, true);
        if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
          AppState.ws.send(JSON.stringify({ type: 'user_transcript', text: text }));
        }
      } else if (!AppState.callActive) {
        showToast('Start the call first by clicking "Try Voice Call"', 'info');
      }
    });
  });
}

async function resetDemoData() {
  try {
    const res = await fetch('/api/reset', { method: 'POST' });
    const data = await res.json();
    showToast(data.message || 'Clinic database reset.', 'success');
    addActivityLog('DATA_RESET', 'Restored doctors and appointments to initial seed.');
  } catch (err) {
    showToast('Failed to reset database.', 'info');
  }
}

function addActivityLog(tag, msg) {
  const feed = DOM.activityFeed;
  if (!feed) return;

  const now = new Date();
  const timeStr = now.toTimeString().split(' ')[0];

  const item = document.createElement('div');
  item.className = 'feed-item';
  item.innerHTML = `
    <span class="feed-time">${timeStr}</span>
    <span class="feed-tag">${escapeHtml(tag)}</span>
    <span class="feed-msg">${escapeHtml(msg)}</span>
  `;

  feed.prepend(item);

  const count = feed.children.length;
  if (DOM.logCounter) DOM.logCounter.textContent = `${count} Events`;
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '&#10003;' : '&#8505;'}</span> <span>${escapeHtml(message)}</span>`;
  DOM.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

function formatPhone(phone) {
  if (!phone) return '';
  const digits = String(phone).replace(/\D/g, '');
  if (digits.length === 10) {
    return `${digits.slice(0, 5)} ${digits.slice(5)}`;
  }
  return digits;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
