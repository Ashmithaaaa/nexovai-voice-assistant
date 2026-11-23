"""
bot.py - Voice AI Pipeline for Doctor Appointment Booking

Built with Pipecat:
  Caller Audio -> VAD (Silero) -> STT (ElevenLabs) -> LLM (Google Gemini) -> TTS (ElevenLabs) -> Caller Audio

Features:
  - RawPCMSerializer: Bi-directional 16kHz PCM audio & control messaging over WebSocket.
  - Tap: Non-intrusive frame processor intercepting live transcripts, speech states, and tool invocations.
  - LLM Function Calling: check_doctor_availability, book_appointment, cancel_appointment.
  - Strict Verification: Caller identified only by spoken phone number; number must be read back & confirmed.
  - Fallback Engine: Gracefully provides simulated conversational flow if API keys are missing.
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
from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
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
- Available Doctors:
  * Dr. Arvind Rao (Cardiology)
  * Dr. Priya Sharma (Dermatology)
  * Dr. Rohan Mehta (Orthopedics)
  * Dr. Ananya Iyer (General Medicine)

CRITICAL RULES:
1. IDENTIFICATION:
   - The caller is identified ONLY by the phone number they say aloud. There is no login, OTP, or caller ID.

2. AVAILABILITY CHECK:
   - When the caller asks about availability (e.g. "Does Dr. Rao have anything tomorrow?"), immediately use the `check_doctor_availability` tool.
   - Read the available slots clearly and offer them to the caller.

3. APPOINTMENT BOOKING:
   - When the caller wants to book an appointment, ask for their phone number if you do not have it.
   - MANDATORY STEP: When the caller states their phone number, YOU MUST READ BACK THE NUMBER (e.g. "Just to confirm, I have your number as 9 8 7 6 5 4 3 2 1 0. Is that correct?").
   - ONLY after the caller explicitly confirms (e.g. "Yes", "Correct", "That's right"), call `book_appointment`.
   - Never book before the caller confirms the phone number.

4. APPOINTMENT CANCELLATION:
   - When the caller wants to cancel, ask for their phone number.
   - MANDATORY STEP: READ BACK the phone number to confirm before calling `cancel_appointment`.
   - ONLY after the caller confirms, call `cancel_appointment`.

5. NATURAL VOICE CONVERSATION:
   - Speak naturally without bullet points, asterisks, or markdown symbols.
   - Be helpful, clear, and reassuring.
"""

GREETING = "Hello! Thank you for calling Aura Health Clinic. I'm your AI receptionist. How can I help you today? You can ask about doctor availability, book an appointment, or cancel one."


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
# 5. Pipeline Runner (Pipecat)
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
    eleven_voice = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()  # Default: Rachel

    # Check if real keys are available
    has_real_keys = bool(gemini_key and eleven_key and not gemini_key.startswith("your_") and not eleven_key.startswith("your_"))

    if not has_real_keys:
        logger.warning("[VoicePipeline] Missing or placeholder API keys. Running in Graceful Simulation Mode.")
        await run_simulated_session(websocket, event_callback)
        return

    logger.info("[VoicePipeline] Initializing Pipecat pipeline with ElevenLabs and Gemini...")

    async with aiohttp.ClientSession() as aiohttp_session:
        # 1. Transport
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

        # 2. VAD
        vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))

        # 3. ElevenLabs STT
        stt = ElevenLabsSTTService(
            api_key=eleven_key,
            aiohttp_session=aiohttp_session,
            sample_rate=16000,
        )

        # 4. Google Gemini LLM
        tools_schema = get_tools_schema()
        llm = GoogleLLMService(
            api_key=gemini_key,
            model="gemini-2.0-flash",
            system_instruction=PROMPT,
        )

        # Wire Tool Handlers
        async def handle_availability(params: FunctionCallParams):
            args = params.arguments
            logger.info(f"[Tool: check_doctor_availability] args: {args}")
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
            logger.info(f"[Tool: book_appointment] args: {args}")
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
            logger.info(f"[Tool: cancel_appointment] args: {args}")
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

        # 5. ElevenLabs TTS
        tts = ElevenLabsTTSService(
            api_key=eleven_key,
            voice_id=eleven_voice,
            model="eleven_flash_v2_5",
            sample_rate=16000,
        )

        # 6. LLM Context & Aggregator Pair
        context = LLMContext(tools=tools_schema)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

        # 7. Tap Frame Processor
        tap = Tap(event_callback)

        # 8. Assemble Pipeline
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

        # Event: Client Connected -> Send initial greeting
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


# ---------------------------------------------------------------------------
# 6. Graceful Simulation Engine (Offline / Demo Fallback)
# ---------------------------------------------------------------------------
async def run_simulated_session(
    websocket: Any,
    event_callback: Callable[[Dict[str, Any]], Awaitable[None]],
):
    """
    Intelligent conversational simulation fallback.
    Ensures 100% testability of all frontend features, audio streams,
    and booking/cancellation logic without failing if keys are omitted.
    """
    logger.info("[Simulation] Starting interactive session...")

    # State tracking for conversational rules
    state = {
        "stage": "idle",
        "pending_action": None,  # "book" or "cancel"
        "pending_doc": "Dr. Arvind Rao",
        "pending_date": "Tomorrow",
        "pending_slot": "10:00 AM",
        "pending_phone": None,
    }

    # Initial greeting
    await asyncio.sleep(0.5)
    await event_callback({"type": "state", "status": "speaking"})
    await event_callback({
        "type": "transcript",
        "role": "agent",
        "text": GREETING,
    })
    await event_callback({"type": "state", "status": "listening"})

    try:
        while True:
            # Receive message from browser
            data = await websocket.receive()
            if "text" in data and data["text"]:
                msg = json.loads(data["text"])
                msg_type = msg.get("type")

                if msg_type == "user_transcript":
                    user_text = msg.get("text", "").strip()
                    if not user_text:
                        continue

                    await event_callback({"type": "state", "status": "thinking"})
                    await asyncio.sleep(0.4)

                    reply_text = await handle_simulated_dialogue(user_text, state, event_callback)

                    await event_callback({"type": "state", "status": "speaking"})
                    await event_callback({
                        "type": "transcript",
                        "role": "agent",
                        "text": reply_text,
                    })
                    await event_callback({"type": "state", "status": "listening"})

            elif "bytes" in data and data["bytes"]:
                # Audio stream received from client mic
                pass

    except Exception as e:
        logger.info(f"[Simulation] Call session finished: {e}")


async def handle_simulated_dialogue(
    user_text: str,
    state: Dict[str, Any],
    event_callback: Callable[[Dict[str, Any]], Awaitable[None]],
) -> str:
    """Conversational state machine replicating strict doctor receptionist rules."""
    text_lower = user_text.lower()
    extracted_phone = clean_phone(user_text)

    # 1. Check if user is confirming pending phone verification
    if state["pending_action"] and state["pending_phone"]:
        is_yes = any(w in text_lower for w in ["yes", "correct", "right", "confirm", "yeah", "yep", "sure"])
        is_no = any(w in text_lower for w in ["no", "wrong", "incorrect", "nope", "change"])

        if is_yes:
            action = state["pending_action"]
            phone = state["pending_phone"]
            state["pending_action"] = None
            state["pending_phone"] = None

            if action == "book":
                res = await book_appointment(
                    patient_phone=phone,
                    doctor_name=state["pending_doc"],
                    date=state["pending_date"],
                    time_slot=state["pending_slot"],
                )
                await event_callback({
                    "type": "tool_event",
                    "action": "book_appointment",
                    "details": res,
                })
                formatted_p = " ".join(phone)
                return f"Thank you! Your appointment with {state['pending_doc']} for {state['pending_date']} at {state['pending_slot']} is officially confirmed under phone number {phone}. We look forward to seeing you!"

            elif action == "cancel":
                res = await cancel_appointment(patient_phone=phone)
                await event_callback({
                    "type": "tool_event",
                    "action": "cancel_appointment",
                    "details": res,
                })
                if res["status"] == "success":
                    return f"Your appointment with {res['doctor']} on {res['date']} at {res['time_slot']} has been successfully cancelled. The slot has been released."
                else:
                    return f"I checked our records, but couldn't find an active appointment under {phone}. Please verify the number and try again."

        elif is_no:
            state["pending_phone"] = None
            return "No problem. Could you please repeat your correct phone number?"

    # 2. Check for Phone Number input when an action is already in progress
    if state["pending_action"] and len(extracted_phone) >= 7:
        state["pending_phone"] = extracted_phone
        spaced_phone = " ".join(extracted_phone)
        action_name = "booking" if state["pending_action"] == "book" else "cancellation"
        return f"Just to confirm, I have your phone number as {spaced_phone}. Is that correct for your {action_name}?"

    # 3. Intent: Cancellation
    if any(w in text_lower for w in ["cancel", "delete", "drop", "reschedule"]):
        state["pending_action"] = "cancel"
        if len(extracted_phone) >= 7:
            state["pending_phone"] = extracted_phone
            spaced_phone = " ".join(extracted_phone)
            return f"I can help you cancel that. Just to confirm, your phone number is {spaced_phone}. Is that correct?"
        return "I can certainly help you cancel your appointment. Could you please tell me the phone number you used when booking?"

    # 4. Intent: Booking
    if any(w in text_lower for w in ["book", "reserve", "schedule", "take the slot", "want that"]):
        state["pending_action"] = "book"
        # Match doctor if mentioned
        for doc in get_doctors_data():
            names = doc["name"].lower().replace("dr.", "").split()
            if any(n in text_lower for n in names):
                state["pending_doc"] = doc["name"]
                break

        # Match slot if mentioned
        if "tomorrow" in text_lower:
            state["pending_date"] = "Tomorrow"
        elif "thursday" in text_lower:
            state["pending_date"] = "Thursday"

        if len(extracted_phone) >= 7:
            state["pending_phone"] = extracted_phone
            spaced_phone = " ".join(extracted_phone)
            return f"I would be glad to book that with {state['pending_doc']} for {state['pending_date']} at {state['pending_slot']}. Just to confirm, your phone number is {spaced_phone}. Is that correct?"
        return f"I would love to book that slot for you with {state['pending_doc']} on {state['pending_date']} at {state['pending_slot']}. What is your phone number?"

    # 5. Intent: Check Availability
    if any(w in text_lower for w in ["available", "availability", "slot", "free", "time", "doctor", "rao", "sharma", "mehta", "iyer"]):
        doc_name = None
        for doc in get_doctors_data():
            names = doc["name"].lower().replace("dr.", "").split()
            if any(n in text_lower for n in names):
                doc_name = doc["name"]
                state["pending_doc"] = doc["name"]
                break

        res = check_doctor_availability(doctor_name=doc_name)
        await event_callback({
            "type": "tool_event",
            "action": "check_doctor_availability",
            "details": res,
        })

        if res["status"] == "success" and res["results"]:
            first_doc = res["results"][0]
            slots = first_doc["available_slots"][:3]
            slots_str = ", ".join(slots)
            return f"{first_doc['doctor']} in {first_doc['specialty']} has the following slots available: {slots_str}. Would you like to book one of these?"
        else:
            return "All our doctors have slots open for Tomorrow and Thursday, including Dr. Rao in Cardiology and Dr. Sharma in Dermatology. Would you like to check a specific doctor?"

    # 6. Default / Fallback Response
    return "I can assist you with checking doctor availability, booking an appointment, or cancelling one. Who would you like to see?"
