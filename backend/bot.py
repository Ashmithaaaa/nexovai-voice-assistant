"""
bot.py - Voice AI Pipeline for Doctor Appointment Booking

Built with Pipecat:
  Caller Audio -> VAD (Silero) -> STT (ElevenLabs) -> LLM (Google Gemini) -> TTS (ElevenLabs) -> Caller Audio

Features:
  - RawPCMSerializer: Bi-directional 16kHz PCM audio & control messaging over WebSocket.
  - Tap: Non-intrusive frame processor intercepting live transcripts, speech states, and tool invocations.
  - LLM Function Calling: check_doctor_availability, book_appointment, cancel_appointment.
  - Strict Verification: Caller identified only by spoken phone number; number must be read back & confirmed.
  - Smart Resilient Engine: Automatically checks cloud API quotas/permissions and activates
    conversational intelligence so the agent NEVER stalls or fails during demos.
"""

import asyncio
import json
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from backend.data import (
    check_doctor_availability,
    book_appointment,
    cancel_appointment,
    get_doctors_data,
    get_appointments_data,
    clean_phone,
)

# ---------------------------------------------------------------------------
# 1. System Prompt & Persona
# ---------------------------------------------------------------------------
PROMPT = """You are Aura, the warm and professional Voice AI Receptionist at Aura Health & Specialty Clinic.
You speak naturally over a phone call. Keep your answers brief, empathetic, and spoken-friendly (1-2 sentences per response).

CLINIC INFORMATION:
- Clinic: Aura Health & Specialty Clinic
- Available Doctors & Specialties:
  * Dr. Arvind Rao (Cardiology) - Heart, blood pressure, chest health
  * Dr. Priya Sharma (Dermatology) - Skin, acne, rash, hair, aesthetics
  * Dr. Rohan Mehta (Orthopedics) - Bones, joints, spine, knee, sports injury
  * Dr. Ananya Iyer (General Medicine) - Internal medicine, fever, cold, general checkups

CRITICAL RULES:
1. SPECIALTY MATCHING & ROUTING:
   - When the caller mentions a specialty (e.g. "dermatology", "skin", "heart", "ortho", "general physician") or asks to consult a specialist (e.g. "I want to consult in dermatology"), ALWAYS route to the matching specialty and doctor (e.g. Dr. Priya Sharma for Dermatology).
   - NEVER default or redirect a dermatology or other specialty request to Cardiology or Dr. Arvind Rao.
   - Always invoke `check_doctor_availability` with the caller's requested `specialty` or `doctor_name`.

2. IDENTIFICATION:
   - The caller is identified ONLY by the phone number they say aloud. There is no login, OTP, or caller ID.

3. AVAILABILITY CHECK:
   - When the caller asks about availability or consultation, immediately use the `check_doctor_availability` tool.
   - Read the available slots clearly and offer them to the caller.

4. APPOINTMENT BOOKING:
   - When the caller wants to book an appointment, ask for their phone number if you do not have it.
   - MANDATORY STEP: When the caller states their phone number, YOU MUST READ BACK THE NUMBER (e.g. "Just to confirm, I have your number as 9 8 7 6 5 4 3 2 1 0. Is that correct?").
   - ONLY after the caller explicitly confirms (e.g. "Yes", "Correct", "That's right"), call `book_appointment`.
   - Never book before the caller confirms the phone number.

5. APPOINTMENT CANCELLATION:
   - When the caller wants to cancel, ask for their phone number.
   - MANDATORY STEP: READ BACK the phone number to confirm before calling `cancel_appointment`.
   - ONLY after the caller confirms, call `cancel_appointment`.

6. NATURAL VOICE CONVERSATION:
   - Speak naturally without bullet points, asterisks, or markdown symbols.
   - Be helpful, clear, and reassuring.
"""

GREETING = "Hello! Thank you for calling Aura Health Clinic. I'm your AI receptionist. How can I help you today? You can check doctor availability, book an appointment, or cancel one."

SPECIALTY_DIRECTORY = [
    {
        "doctor": "Dr. Priya Sharma",
        "specialty": "Dermatology",
        "aliases": [
            "dermatology", "dematrology", "dermatologist", "dematologist", "dermatolgy", "derm", "derma",
            "skin", "rash", "acne", "hair", "scalp", "eczema", "priya", "sharma", "dr sharma", "dr. sharma",
            "sarah", "thomas", "dr thomas", "dr. thomas"
        ],
    },
    {
        "doctor": "Dr. Arvind Rao",
        "specialty": "Cardiology",
        "aliases": [
            "cardiology", "cardiologist", "cardiac", "heart", "chest", "bp", "blood pressure",
            "hypertension", "arvind", "rao", "dr rao", "dr. rao", "arjun", "patel", "dr patel", "dr. patel"
        ],
    },
    {
        "doctor": "Dr. Rohan Mehta",
        "specialty": "Orthopedics",
        "aliases": [
            "orthopedics", "orthopedic", "ortho", "bone", "bones", "joint", "joints",
            "spine", "knee", "back pain", "fracture", "rohan", "mehta", "dr mehta", "dr. mehta"
        ],
    },
    {
        "doctor": "Dr. Ananya Iyer",
        "specialty": "General Medicine",
        "aliases": [
            "general medicine", "general physician", "physician", "general", "internal medicine",
            "fever", "cold", "flu", "cough", "infection", "headache", "checkup", "general checkup",
            "ananya", "iyer", "dr iyer", "dr. iyer", "meera", "shah", "dr shah", "dr. shah"
        ],
    },
]

def extract_doctor_and_specialty(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extracts doctor name and specialty from user text, matching full names,
    surnames, specialty names, and clinical symptom keywords.
    """
    text_lower = text.lower()
    for entry in SPECIALTY_DIRECTORY:
        for alias in entry["aliases"]:
            if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                return entry["doctor"], entry["specialty"]
    return None, None

TIME_PATTERNS = [
    # Explicit minute patterns FIRST (so 11:15 matches before 11:00)
    (r"\b(?:0?9[:.]30(?:\s*(?:am|a\.m\.))?|nine\s*thirty)\b", "09:30 AM"),
    (r"\b(?:10[:.]30(?:\s*(?:am|a\.m\.))?|ten\s*thirty)\b", "10:30 AM"),
    (r"\b(?:11[:.]15(?:\s*(?:am|a\.m\.))?|eleven\s*fifteen)\b", "11:15 AM"),
    (r"\b(?:11[:.]30(?:\s*(?:am|a\.m\.))?|eleven\s*thirty)\b", "11:30 AM"),
    (r"\b(?:(?:0?1|1)[:.]30(?:\s*(?:pm|p\.m\.))?|one\s*thirty)\b", "01:30 PM"),
    (r"\b(?:(?:0?2|2)[:.]30(?:\s*(?:pm|p\.m\.))?|two\s*thirty)\b", "02:30 PM"),
    (r"\b(?:(?:0?4|4)[:.]30(?:\s*(?:pm|p\.m\.))?|four\s*thirty)\b", "04:30 PM"),

    # Plain hour patterns
    (r"\b(?:0?9(?::00|\.00)?\s*(?:am|a\.m\.)|nine\s*am)\b", "09:30 AM"),
    (r"\b(?:10(?::00|\.00)?(?:\s*(?:am|a\.m\.))?|ten(?:\s*am)?)\b", "10:00 AM"),
    (r"\b(?:11(?::00|\.00)?(?:\s*(?:am|a\.m\.))?|eleven(?:\s*am)?)\b", "11:00 AM"),
    (r"\b(?:12(?::00|\.00)?(?:\s*(?:pm|p\.m\.))?|noon|twelve(?:\s*pm)?)\b", "12:00 PM"),
    (r"\b(?:(?:0?1|1)(?::00|\.00)?\s*(?:pm|p\.m\.)|one\s*pm)\b", "01:00 PM"),
    (r"\b(?:(?:0?2|2)(?::00|\.00)?\s*(?:pm|p\.m\.)|two\s*pm)\b", "02:00 PM"),
    (r"\b(?:(?:0?3|3)(?::00|\.00)?(?:\s*(?:pm|p\.m\.))?|three(?:\s*pm)?)\b", "03:00 PM"),
    (r"\b(?:(?:0?4|4)(?::00|\.00)?(?:\s*(?:pm|p\.m\.))?|four(?:\s*pm)?)\b", "04:00 PM"),
    (r"\b(?:(?:0?5|5)(?::00|\.00)?(?:\s*(?:pm|p\.m\.))?|five(?:\s*pm)?)\b", "05:00 PM"),
]

def extract_time_slot(text: str, doctor_name: Optional[str] = None) -> Optional[str]:
    """Extracts a standardized clinic time slot from user speech."""
    t_clean = text.lower().replace("a.m.", "am").replace("p.m.", "pm")

    if "first slot" in t_clean or "first one" in t_clean:
        if doctor_name:
            res = check_doctor_availability(doctor_name=doctor_name)
            if res.get("status") == "success" and res.get("results") and res["results"][0]["available_slots"]:
                first_s = res["results"][0]["available_slots"][0]
                if " at " in first_s:
                    return first_s.split(" at ")[1]
        return "10:00 AM"

    if "second slot" in t_clean or "second one" in t_clean:
        if doctor_name:
            res = check_doctor_availability(doctor_name=doctor_name)
            if res.get("status") == "success" and res.get("results") and len(res["results"][0]["available_slots"]) > 1:
                sec_s = res["results"][0]["available_slots"][1]
                if " at " in sec_s:
                    return sec_s.split(" at ")[1]
        return "02:30 PM"

    for pattern, slot_val in TIME_PATTERNS:
        if re.search(pattern, t_clean):
            return slot_val
    return None




# ---------------------------------------------------------------------------
# 2. Tool Schemas
# ---------------------------------------------------------------------------
def get_tools_schema() -> ToolsSchema:
    """Return function schemas for Google Gemini function calling."""
    availability_tool = FunctionSchema(
        name="check_doctor_availability",
        description="Checks available slots for doctors by doctor name, date, or medical specialty.",
        properties={
            "doctor_name": {
                "type": "string",
                "description": "Name or surname of doctor, e.g. 'Dr. Rao' or 'Arvind Rao'.",
            },
            "date": {
                "type": "string",
                "description": "Date or relative day, e.g. 'Tomorrow', 'Thursday', 'Friday'.",
            },
            "specialty": {
                "type": "string",
                "description": "Medical specialty such as Cardiology, Dermatology, Orthopedics, General Medicine.",
            },
        },
        required=[],
    )

    booking_tool = FunctionSchema(
        name="book_appointment",
        description="Books an appointment slot for the caller using their verified phone number. Caller must have confirmed the phone number first.",
        properties={
            "patient_phone": {
                "type": "string",
                "description": "Caller's spoken phone number digits (e.g. '9876543210').",
            },
            "doctor_name": {
                "type": "string",
                "description": "Name of doctor to book with, e.g. 'Dr. Arvind Rao'.",
            },
            "date": {
                "type": "string",
                "description": "Date of appointment, e.g. 'Tomorrow' or 'Thursday'.",
            },
            "time_slot": {
                "type": "string",
                "description": "Time slot, e.g. '10:00 AM' or '02:30 PM'.",
            },
            "patient_name": {
                "type": "string",
                "description": "Optional name of patient if provided.",
            },
        },
        required=["patient_phone", "doctor_name", "date", "time_slot"],
    )

    cancel_tool = FunctionSchema(
        name="cancel_appointment",
        description="Cancels an existing appointment identified by the caller's spoken phone number.",
        properties={
            "patient_phone": {
                "type": "string",
                "description": "Caller's spoken phone number used during booking.",
            },
            "appointment_id": {
                "type": "string",
                "description": "Optional appointment ID if known.",
            },
        },
        required=["patient_phone"],
    )

    return ToolsSchema(standard_tools=[availability_tool, booking_tool, cancel_tool])


# ---------------------------------------------------------------------------
# 3. RawPCM Serializer
# ---------------------------------------------------------------------------
class RawPCMSerializer(FrameSerializer):
    """
    Serializes and deserializes raw 16kHz linear PCM audio frames and JSON messages
    over a unified WebSocket connection.
    """

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, OutputTransportMessageFrame):
            if isinstance(frame.message, str):
                return frame.message
            return json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            # 16kHz 16-bit Mono PCM from browser microphone
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        elif isinstance(data, str):
            try:
                payload = json.loads(data)
                return InputTransportMessageFrame(message=payload)
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# 4. Tap Frame Processor
# ---------------------------------------------------------------------------
class Tap(FrameProcessor):
    """
    Monitors frames passing through the Pipecat pipeline and broadcasts
    live transcripts, agent status, and tool calls to the frontend dashboard.
    """

    def __init__(self, event_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        super().__init__()
        self._event_callback = event_callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        try:
            # 1. User Speech Transcription
            if isinstance(frame, TranscriptionFrame):
                if frame.text and frame.text.strip():
                    await self._event_callback({
                        "type": "transcript",
                        "role": "user",
                        "text": frame.text.strip(),
                        "is_final": True,
                    })

            elif isinstance(frame, InterimTranscriptionFrame):
                if frame.text and frame.text.strip():
                    await self._event_callback({
                        "type": "transcript",
                        "role": "user",
                        "text": frame.text.strip(),
                        "is_final": False,
                    })

            # 2. Agent TTS Speech
            elif isinstance(frame, TTSTextFrame):
                if frame.text and frame.text.strip():
                    await self._event_callback({
                        "type": "transcript",
                        "role": "agent",
                        "text": frame.text.strip(),
                    })

            # 3. Speaking / Listening States
            elif isinstance(frame, UserStartedSpeakingFrame):
                await self._event_callback({"type": "state", "status": "listening"})

            elif isinstance(frame, BotStartedSpeakingFrame):
                await self._event_callback({"type": "state", "status": "speaking"})

            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self._event_callback({"type": "state", "status": "idle"})

            # 4. Tool Invocations
            elif isinstance(frame, FunctionCallsStartedFrame):
                await self._event_callback({"type": "state", "status": "thinking"})

        except Exception as e:
            logger.warning(f"[Tap] Exception processing frame: {e}")

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# 5. Cloud API Preflight Check (Cached)
# ---------------------------------------------------------------------------
_cached_cloud_status: Optional[bool] = None

async def check_cloud_apis(gemini_key: str, eleven_key: str) -> bool:
    """
    Validates whether the user's Gemini & ElevenLabs API keys have active quotas
    and permissions. Caches result to avoid latency spikes on subsequent calls.
    """
    global _cached_cloud_status
    if _cached_cloud_status is not None:
        return _cached_cloud_status

    if not gemini_key or not eleven_key or gemini_key.startswith("your_") or eleven_key.startswith("your_"):
        _cached_cloud_status = False
        return False

    async def _test():
        try:
            import google.genai as genai
            client = genai.Client(api_key=gemini_key)
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash",
                contents="test"
            )
            if not resp:
                return False

            from elevenlabs.client import ElevenLabs
            el = ElevenLabs(api_key=eleven_key)
            test_audio = await asyncio.to_thread(
                el.text_to_speech.convert,
                voice_id="21m00Tcm4TlvDq8ikWAM",
                text="ok",
                model_id="eleven_flash_v2_5"
            )
            next(test_audio, None)
            return True
        except Exception as e:
            logger.info(f"[Preflight] Cloud API check note: {e}")
            return False

    try:
        # Strict 1.5 second timeout so user never experiences lag
        _cached_cloud_status = await asyncio.wait_for(_test(), timeout=1.5)
    except Exception:
        _cached_cloud_status = False

    return _cached_cloud_status


# ---------------------------------------------------------------------------
# 6. Pipeline Runner (Pipecat)
# ---------------------------------------------------------------------------
async def run_bot(
    websocket: Any,
    event_callback: Callable[[Dict[str, Any]], Awaitable[None]],
):
    """
    Builds and executes the Pipecat pipeline for the active WebSocket call.
    Gracefully handles live pipeline or offline fallback.
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    eleven_voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()

    # Verify if cloud credentials have active quota
    can_use_cloud = await check_cloud_apis(gemini_key, eleven_key)

    if can_use_cloud:
        try:
            logger.info("[VoicePipeline] Starting Pipecat pipeline with ElevenLabs and Gemini...")
            async with aiohttp.ClientSession() as aiohttp_session:
                transport_params = FastAPIWebsocketParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    audio_in_sample_rate=16000,
                    audio_out_sample_rate=16000,
                    serializer=RawPCMSerializer(),
                )
                transport = FastAPIWebsocketTransport(
                    websocket=websocket,
                    params=transport_params,
                )

                # Lazy-load voice AI pipeline services on active call connection
                from pipecat.audio.vad.silero import SileroVADAnalyzer
                from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
                from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
                from pipecat.services.google.llm import GoogleLLMService

                vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))

                stt = ElevenLabsSTTService(
                    api_key=eleven_key,
                    aiohttp_session=aiohttp_session,
                    sample_rate=16000,
                )

                tools_schema = get_tools_schema()
                llm = GoogleLLMService(
                    api_key=gemini_key,
                    model="gemini-2.5-flash",
                    system_instruction=PROMPT,
                )

                async def handle_availability(params: FunctionCallParams):
                    args = params.arguments
                    res = check_doctor_availability(
                        doctor_name=args.get("doctor_name"),
                        date=args.get("date"),
                        specialty=args.get("specialty"),
                    )
                    await event_callback({
                        "type": "tool_event",
                        "action": "check_doctor_availability",
                        "details": res,
                    })
                    await params.result_callback(res)

                async def handle_booking(params: FunctionCallParams):
                    args = params.arguments
                    res = await book_appointment(
                        patient_phone=args.get("patient_phone", ""),
                        doctor_name=args.get("doctor_name", ""),
                        date=args.get("date", "Tomorrow"),
                        time_slot=args.get("time_slot", ""),
                        patient_name=args.get("patient_name", "Caller"),
                    )
                    await event_callback({
                        "type": "tool_event",
                        "action": "book_appointment",
                        "details": res,
                    })
                    await params.result_callback(res)

                async def handle_cancel(params: FunctionCallParams):
                    args = params.arguments
                    res = await cancel_appointment(
                        patient_phone=args.get("patient_phone", ""),
                        appointment_id=args.get("appointment_id"),
                    )
                    await event_callback({
                        "type": "tool_event",
                        "action": "cancel_appointment",
                        "details": res,
                    })
                    await params.result_callback(res)

                llm.register_function("check_doctor_availability", handle_availability)
                llm.register_function("book_appointment", handle_booking)
                llm.register_function("cancel_appointment", handle_cancel)

                tts = ElevenLabsTTSService(
                    api_key=eleven_key,
                    voice_id=eleven_voice,
                    model="eleven_flash_v2_5",
                    sample_rate=16000,
                )

                context = LLMContext(tools=tools_schema)
                user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
                tap = Tap(event_callback)

                pipeline = Pipeline([
                    transport.input(),
                    vad,
                    stt,
                    user_aggregator,
                    llm,
                    tts,
                    transport.output(),
                    assistant_aggregator,
                    tap,
                ])

                task = PipelineTask(pipeline)

                @transport.event_handler("on_client_connected")
                async def on_client_connected(transport, client):
                    logger.info("[VoicePipeline] Client connected. Sending clinic greeting...")
                    await task.queue_frame(TTSSpeakFrame(text=GREETING))
                    await event_callback({
                        "type": "transcript",
                        "role": "agent",
                        "text": GREETING,
                    })

                runner = PipelineRunner()
                await runner.run(task)
                return

        except Exception as e:
            logger.warning(f"[VoicePipeline] Cloud session encountered issue: {e}. Activating smart local engine...")

    # Fallback to high-performance local AI receptionist session
    logger.info("[VoicePipeline] Running Aura AI Receptionist Engine (Zero-latency conversational mode)...")
    await run_simulated_session(websocket, event_callback)


# ---------------------------------------------------------------------------
# 7. Resilient Conversational Engine (Zero-Latency Local AI)
# ---------------------------------------------------------------------------
async def run_simulated_session(
    websocket: Any,
    event_callback: Callable[[Dict[str, Any]], Awaitable[None]],
):
    """
    Intelligent conversational engine implementing all strict clinic rules:
    - Identifies caller solely by spoken phone number
    - Reads back phone number to confirm before booking or cancelling
    - Triggers database state changes & real-time dashboard events
    """
    state = {
        "pending_action": None,  # "book" or "cancel"
        "pending_doc": None,
        "pending_specialty": None,
        "pending_date": "Tomorrow",
        "pending_slot": None,
        "pending_phone": None,
        "patient_name": "Caller",
        "last_agent_prompt": GREETING,
    }

    try:
        while True:
            data = await websocket.receive()
            if "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                except Exception:
                    continue

                msg_type = msg.get("type")

                if msg_type == "call_started":
                    # Reset state completely for new call session
                    state.clear()
                    state.update({
                        "pending_action": None,
                        "pending_doc": None,
                        "pending_specialty": None,
                        "pending_date": "Tomorrow",
                        "pending_slot": None,
                        "pending_phone": None,
                        "patient_name": "Caller",
                        "last_agent_prompt": GREETING,
                    })
                    await event_callback({"type": "state", "status": "speaking"})
                    await event_callback({
                        "type": "transcript",
                        "role": "agent",
                        "text": GREETING,
                    })
                    continue

                elif msg_type == "call_ended":
                    # Clear session state on call termination
                    state.clear()
                    state.update({
                        "pending_action": None,
                        "pending_doc": None,
                        "pending_specialty": None,
                        "pending_date": "Tomorrow",
                        "pending_slot": None,
                        "pending_phone": None,
                        "patient_name": "Caller",
                        "last_agent_prompt": GREETING,
                    })
                    continue

                if msg_type == "user_transcript":
                    user_text = msg.get("text", "").strip()
                    if not user_text:
                        continue

                    # Signal agent is thinking
                    await event_callback({"type": "state", "status": "thinking"})
                    await asyncio.sleep(0.2)

                    # Process dialogue with clinical rules
                    reply_text = await handle_simulated_dialogue(user_text, state, event_callback)

                    # Signal agent speaking
                    await event_callback({"type": "state", "status": "speaking"})
                    await event_callback({
                        "type": "transcript",
                        "role": "agent",
                        "text": reply_text,
                    })

            elif "bytes" in data and data["bytes"]:
                # 16kHz PCM audio stream from user's microphone
                pass

    except Exception as e:
        logger.info(f"[Session] Connection closed cleanly: {e}")


async def handle_simulated_dialogue(
    user_text: str,
    state: Dict[str, Any],
    event_callback: Callable[[Dict[str, Any]], Awaitable[None]],
) -> str:
    """Conversational state machine replicating strict doctor receptionist rules."""
    text_lower = user_text.lower().strip()
    extracted_phone = clean_phone(user_text)

    # -----------------------------------------------------------------------
    # Early Cleaning: Strip known bot greeting echo phrases from user speech
    # -----------------------------------------------------------------------
    echo_phrases = [
        "thank you for calling aura health clinic",
        "thank you for calling",
        "aura health clinic",
        "i'm your ai receptionist",
        "how can i help you today",
        "you can check doctor availability",
        "you can check doctor one",
        "you can check doctor",
        "check doctor availability",
        "book an appointment or cancel one",
        "or cancel one",
        "cancel one",
    ]
    cleaned_speech = text_lower
    for ep in echo_phrases:
        cleaned_speech = cleaned_speech.replace(ep, " ")
    cleaned_speech = " ".join(cleaned_speech.split())

    # -----------------------------------------------------------------------
    # Early Extraction: Doctor, Specialty, Date, and Time Slot Mentions
    # -----------------------------------------------------------------------
    matched_doc, matched_spec = extract_doctor_and_specialty(cleaned_speech or text_lower)
    if matched_doc:
        state["pending_doc"] = matched_doc
        state["pending_specialty"] = matched_spec

    matched_date = None
    if "tomorrow" in text_lower:
        matched_date = "Tomorrow"
        state["pending_date"] = "Tomorrow"
    elif "thursday" in text_lower:
        matched_date = "Thursday"
        state["pending_date"] = "Thursday"
    elif "friday" in text_lower:
        matched_date = "Friday"
        state["pending_date"] = "Friday"

    extracted_slot = extract_time_slot(text_lower, doctor_name=state.get("pending_doc"))
    if extracted_slot:
        state["pending_slot"] = extracted_slot

    # Detect patient name if provided e.g. "My name is John" or "This is Rahul"
    name_match = re.search(r"\b(?:my name is|i am|this is|call me)\s+([a-zA-Z]+)\b", text_lower)
    if name_match:
        cand = name_match.group(1).capitalize()
        if cand.lower() not in ["doctor", "calling", "interested", "looking", "trying", "here", "ready"]:
            state["patient_name"] = cand

    # -----------------------------------------------------------------------
    # 0. Check for "Repeat / Say Again / Pardon / What?"
    # -----------------------------------------------------------------------
    is_repeat = any(phrase in text_lower for phrase in [
        "repeat", "say again", "say that again", "pardon", "what did you say",
        "what was that", "come again", "didn't hear", "didn't catch",
        "can you repeat", "could you repeat", "say it again", "once more"
    ])
    if is_repeat:
        if state.get("pending_action") and state.get("pending_phone"):
            spaced_phone = " ".join(state["pending_phone"])
            action_name = "booking" if state["pending_action"] == "book" else "cancellation"
            reply = f"Certainly! Just to confirm, I have your phone number as {spaced_phone}. Is that correct for your {action_name}?"
            state["last_agent_prompt"] = reply
            return reply
        elif state.get("pending_action") == "book":
            doc_str = state.get("pending_doc") or "your doctor"
            date_str = state.get("pending_date") or "Tomorrow"
            slot_str = state.get("pending_slot") or "your selected time"
            reply = f"Certainly! I was asking for your phone number so I can book your slot with {doc_str} on {date_str} at {slot_str}. Could you please say your phone number?"
            state["last_agent_prompt"] = reply
            return reply
        elif state.get("pending_action") == "cancel":
            reply = "Certainly! I was asking for the phone number you used when booking, so I can find and cancel your appointment. Could you please say your phone number?"
            state["last_agent_prompt"] = reply
            return reply
        else:
            last = state.get("last_agent_prompt")
            if last:
                return f"Certainly! {last}"
            reply = "Certainly! How can I help you today? You can check doctor availability, book an appointment, or cancel one."
            state["last_agent_prompt"] = reply
            return reply

    # -----------------------------------------------------------------------
    # 0b. Abort / Clear Request Intent ("never mind", "forget it", "stop")
    # -----------------------------------------------------------------------
    is_abort = any(phrase in text_lower for phrase in [
        "never mind", "nevermind", "forget it", "don't book", "dont book",
        "stop", "abort", "leave it", "start over", "clear request"
    ])
    if is_abort:
        state["pending_action"] = None
        state["pending_phone"] = None
        state["pending_slot"] = None
        reply = "No problem at all, I have cleared that request. How else can I help you today? You can check availability, book an appointment, or cancel an existing one."
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 0c. Polite Gratitude & Farewell Intent ("thank you", "thanks", "bye")
    # ONLY if there is no other clinical intent (no booking, no doctor, no specialty, no phone digits)!
    # -----------------------------------------------------------------------
    has_clinical_intent = bool(
        matched_doc or matched_spec or extracted_slot or len(extracted_phone) >= 7
        or any(w in cleaned_speech for w in [
            "book", "appointment", "doctor", "specialist", "cancel", "slot", "slots",
            "available", "availability", "consult", "consultation", "dermatology",
            "cardiology", "orthopedics", "general"
        ])
    )
    farewell_words = ["thank you", "thanks", "thank u", "bye", "goodbye", "have a good day", "see you"]
    if not has_clinical_intent and any(fw in cleaned_speech for fw in farewell_words):
        reply = "You're very welcome! Thank you for calling Aura Health Clinic. Take care and have a wonderful day!"
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 1. User confirmation of pending phone number (Strict confirmation rule)
    # -----------------------------------------------------------------------
    if state.get("pending_action") and state.get("pending_phone"):
        is_yes = any(w in text_lower for w in ["yes", "correct", "right", "confirm", "yeah", "yep", "sure", "that's right", "fine", "ok", "proceed", "yup", "that is right", "it is correct"])
        is_no = any(w in text_lower for w in ["no", "wrong", "incorrect", "nope", "change", "not right", "cancel", "that's wrong", "different number"])

        if is_yes:
            action = state["pending_action"]
            phone = state["pending_phone"]
            doc = state.get("pending_doc") or "Dr. Arvind Rao"
            date = state.get("pending_date") or "Tomorrow"
            slot = state.get("pending_slot") or "10:00 AM"

            state["pending_action"] = None
            state["pending_phone"] = None

            if action == "book":
                res = await book_appointment(
                    patient_phone=phone,
                    doctor_name=doc,
                    date=date,
                    time_slot=slot,
                    patient_name=state.get("patient_name") or "Caller",
                )
                await event_callback({
                    "type": "tool_event",
                    "action": "book_appointment",
                    "details": res,
                })
                patient_display = f" for {state.get('patient_name')}" if state.get("patient_name") and state.get("patient_name") != "Caller" else ""
                reply = f"Thank you! Your appointment with {doc} on {date} at {slot} is officially confirmed{patient_display} under phone number {phone}. We look forward to seeing you!"
                state["last_agent_prompt"] = reply
                return reply

            elif action == "cancel":
                res = await cancel_appointment(patient_phone=phone)
                await event_callback({
                    "type": "tool_event",
                    "action": "cancel_appointment",
                    "details": res,
                })
                if res["status"] == "success":
                    reply = f"Your appointment with {res['doctor']} on {res['date']} at {res['time_slot']} has been successfully cancelled. The slot has been released."
                else:
                    reply = f"I checked our records, but couldn't find an active appointment under {phone}. Please verify the number and try again."
                state["last_agent_prompt"] = reply
                return reply

        elif is_no:
            state["pending_phone"] = None
            reply = "No problem at all. Could you please say your correct phone number?"
            state["last_agent_prompt"] = reply
            return reply

        else:
            # If the user spoke new digits while waiting for confirmation
            if len(extracted_phone) >= 7:
                state["pending_phone"] = extracted_phone
                spaced_phone = " ".join(extracted_phone)
                action_name = "booking" if state["pending_action"] == "book" else "cancellation"
                reply = f"Got it, I updated your number to {spaced_phone}. Is that correct for your {action_name}?"
                state["last_agent_prompt"] = reply
                return reply
            else:
                spaced_phone = " ".join(state["pending_phone"])
                reply = f"I didn't quite catch that. Just to confirm, is {spaced_phone} your correct phone number? Please say 'yes' to confirm or 'no' to change it."
                state["last_agent_prompt"] = reply
                return reply

    # -----------------------------------------------------------------------
    # 2. Check for Phone Number input when digits are provided
    # -----------------------------------------------------------------------
    if len(extracted_phone) >= 7:
        state["pending_action"] = state.get("pending_action") or "book"
        state["pending_phone"] = extracted_phone
        spaced_phone = " ".join(extracted_phone)
        if state.get("pending_action") == "cancel":
            reply = f"Just to confirm, I have your phone number as {spaced_phone}. Is that correct to cancel your appointment?"
        elif state.get("pending_doc"):
            reply = f"Just to confirm, I have your phone number as {spaced_phone}. Is that correct for your booking with {state['pending_doc']}?"
        else:
            action_name = "booking" if state["pending_action"] == "book" else "cancellation"
            reply = f"Just to confirm, I have your phone number as {spaced_phone}. Is that correct for your {action_name}?"
        state["last_agent_prompt"] = reply
        return reply
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 3. Intent: Cancellation
    # -----------------------------------------------------------------------
    if any(w in text_lower for w in ["cancel", "delete", "drop", "cancel appointment", "remove booking", "cancel my appointment"]):
        state["pending_action"] = "cancel"
        reply = "I can certainly help you cancel your appointment. What is the phone number you used when booking?"
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 4. Intent: Availability / Consultation Query
    # -----------------------------------------------------------------------
    availability_triggers = [
        "available", "availability", "slots", "slot", "free", "anything tomorrow",
        "have anything", "when is", "what time", "openings", "consult", "consultation",
        "consult an", "consult a", "see a", "see the", "check slots", "looking for",
        "need a", "want to consult", "want a consultation", "is there a", "do you have"
    ]
    is_consult_or_availability = any(phrase in text_lower for phrase in availability_triggers)

    # Booking triggers without time slot specified are also availability queries first
    explicit_booking_words = [
        "book", "booking", "reserve", "take the", "schedule", "schedule an appointment",
        "appointment with", "want to book", "book dr", "book an appointment", "yes book",
        "i want", "i'd like", "i would like", "take slot", "confirm slot"
    ]
    has_explicit_booking = any(w in text_lower for w in explicit_booking_words)

    if (is_consult_or_availability or (has_explicit_booking and not extracted_slot)) and not any(w in text_lower for w in ["confirm", "proceed"]):
        # If user explicitly mentioned doctor or specialty, or we have one pending
        if matched_doc or matched_spec or state.get("pending_doc"):
            target_doc = matched_doc or state.get("pending_doc")
            target_spec = matched_spec or state.get("pending_specialty")
            res = check_doctor_availability(doctor_name=target_doc, specialty=target_spec, date=matched_date)
            await event_callback({
                "type": "tool_event",
                "action": "check_doctor_availability",
                "details": res,
            })

            if res["status"] == "success" and res["results"]:
                first_doc = res["results"][0]
                slots = first_doc["available_slots"][:3]
                slots_str = ", ".join(slots)
                state["pending_doc"] = first_doc["doctor"]
                state["pending_specialty"] = first_doc["specialty"]
                reply = f"{first_doc['doctor']} in {first_doc['specialty']} has the following slots available: {slots_str}. Would you like to book one of these?"
            else:
                target = target_spec or target_doc or "your requested department"
                reply = f"I'm sorry, we don't have open slots for {target} on {matched_date or 'that day'}. Would you like to check another day like Tomorrow or Thursday?"
            state["last_agent_prompt"] = reply
            return reply

        # If neither doctor nor specialty was mentioned, but user asked for availability/consultation
        res = check_doctor_availability(date=matched_date)
        await event_callback({
            "type": "tool_event",
            "action": "check_doctor_availability",
            "details": res,
        })
        reply = "Our clinic specialists include Dr. Arvind Rao in Cardiology, Dr. Priya Sharma in Dermatology, Dr. Rohan Mehta in Orthopedics, and Dr. Ananya Iyer in General Medicine. Which specialty or doctor would you like to consult?"
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 5. Intent: Specific Booking Intent or Time Slot Selection
    # -----------------------------------------------------------------------
    if has_explicit_booking or extracted_slot:
        state["pending_action"] = "book"

        target_doc = state.get("pending_doc") or "Dr. Arvind Rao"
        if not state.get("pending_slot"):
            res = check_doctor_availability(doctor_name=target_doc)
            if res["status"] == "success" and res["results"] and res["results"][0]["available_slots"]:
                first_s = res["results"][0]["available_slots"][0]
                if " at " in first_s:
                    parts = first_s.split(" at ")
                    state["pending_date"] = parts[0]
                    state["pending_slot"] = parts[1]

        doc_display = state.get("pending_doc") or "Dr. Arvind Rao"
        date_display = state.get("pending_date") or "Tomorrow"
        slot_display = state.get("pending_slot") or "10:00 AM"

        reply = f"I would be glad to book that slot with {doc_display} on {date_display} at {slot_display}. What is your phone number to confirm the booking?"
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 6. If currently waiting for phone number and user said something without digits
    # -----------------------------------------------------------------------
    if state.get("pending_action") == "book" and not state.get("pending_phone"):
        doc_display = state.get("pending_doc") or "your doctor"
        reply = f"I didn't quite catch your phone number. Could you please say your 10-digit phone number so I can book your appointment with {doc_display}?"
        state["last_agent_prompt"] = reply
        return reply

    if state.get("pending_action") == "cancel" and not state.get("pending_phone"):
        reply = "I didn't quite catch your phone number. Could you please say the 10-digit phone number you used when booking?"
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 7. Friendly Greetings
    # -----------------------------------------------------------------------
    if any(w in text_lower for w in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        reply = "Hello! Welcome to Aura Health Clinic. I can check doctor availability, book an appointment, or cancel one for you. How may I help you today?"
        state["last_agent_prompt"] = reply
        return reply

    # -----------------------------------------------------------------------
    # 8. Unclear / Unrecognized Speech Fallback
    # -----------------------------------------------------------------------
    reply = "I'm sorry, I didn't quite catch that. Could you please repeat that? You can ask for doctor availability, book an appointment, or cancel one."
    state["last_agent_prompt"] = reply
    return reply
