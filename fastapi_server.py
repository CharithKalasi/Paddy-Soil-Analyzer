import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from predict import phase1_predict, phase2_predict


load_dotenv()

app = FastAPI(title="KrishiLink Mobile Bridge API", version="1.0.0")

TEST_PHASE1_INPUT = {
    "N": 80.0,
    "P": 50.0,
    "K": 50.0,
    "ph": 7.0,
    "EC_uS_cm": 1240.0,
}
TEST_PHASE2_INPUT = {
    "ORP_mV": 0.0,
}
TEST_ESP_DEVICE_ID = "test"

# Allow Flutter app calls from mobile and emulator/web debug sessions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Phase1Request(BaseModel):
    N: float = Field(..., ge=0, le=100)
    P: float = Field(..., ge=0, le=70)
    K: float = Field(..., ge=0, le=55)
    ph: float = Field(..., ge=3.5, le=9.0)
    EC_uS_cm: float = Field(..., ge=0, le=3500)
    model_name: str = "llama-3.1-8b-instant"


class Phase2Request(BaseModel):
    ORP_mV: float = Field(..., ge=-350, le=350)
    model_name: str = "llama-3.1-8b-instant"


class FollowupRequest(BaseModel):
    session_id: str
    message: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionState(BaseModel):
    phase: Literal["phase1", "phase2"]
    context_text: str
    model_name: str
    chat_messages: List[ChatMessage] = Field(default_factory=list)


SESSIONS: Dict[str, SessionState] = {}
ESP_RECORDS = deque(maxlen=500)


class ESPIngestRequest(BaseModel):
    device_id: str = "esp-device-1"
    timestamp: Optional[str] = None
    N: Optional[float] = Field(default=None, ge=0, le=100)
    P: Optional[float] = Field(default=None, ge=0, le=70)
    K: Optional[float] = Field(default=None, ge=0, le=55)
    ph: Optional[float] = Field(default=None, ge=3.5, le=9.0)
    EC_uS_cm: Optional[float] = Field(default=None, ge=0, le=3500)
    ORP_mV: Optional[float] = Field(default=None, ge=-350, le=350)


def _make_iso_timestamp(value: Optional[str]) -> str:
    if value and value.strip():
        return value
    return datetime.now(timezone.utc).isoformat()


def _phase1_from_esp(payload: ESPIngestRequest) -> Optional[dict]:
    if None in (payload.N, payload.P, payload.K, payload.ph, payload.EC_uS_cm):
        return None

    result = phase1_predict(
        N=payload.N,
        P=payload.P,
        K=payload.K,
        ph=payload.ph,
        EC=payload.EC_uS_cm,
    )
    return {
        "sensor_data": {
            "N": payload.N,
            "P": payload.P,
            "K": payload.K,
            "ph": payload.ph,
            "EC_uS_cm": payload.EC_uS_cm,
        },
        "recommended_outputs": flatten_recommendations(result),
    }


def _phase2_from_esp(payload: ESPIngestRequest) -> Optional[dict]:
    if payload.ORP_mV is None:
        return None

    result = phase2_predict(ORP=payload.ORP_mV)
    return {
        "sensor_data": {
            "ORP_mV": payload.ORP_mV,
        },
        "recommended_outputs": flatten_recommendations(result),
    }


def flatten_recommendations(result: dict) -> dict:
    flat_result = {}
    for section, values in result.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat_result[key] = round(float(value), 2)
        elif isinstance(values, (int, float)):
            flat_result[section] = round(float(values), 2)
        else:
            flat_result[section] = values
    return flat_result


def format_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def build_phase1_context(sensor_input: dict, recommendations: dict) -> str:
    lines = [
        "Phase 1 sensor input:",
        f"- N: {sensor_input['N']}",
        f"- P: {sensor_input['P']}",
        f"- K: {sensor_input['K']}",
        f"- ph: {sensor_input['ph']}",
        f"- EC_uS_cm: {sensor_input['EC_uS_cm']}",
        "",
        "Model recommendations:",
    ]
    for key, value in recommendations.items():
        lines.append(f"- {key}: {format_value(value)}")
    return "\n".join(lines)


def build_phase2_context(sensor_input: dict, recommendations: dict) -> str:
    lines = [
        "Phase 2 sensor input:",
        f"- ORP_mV: {sensor_input['ORP_mV']}",
        "",
        "Model recommendations:",
    ]
    for key, value in recommendations.items():
        lines.append(f"- {key}: {format_value(value)}")
    return "\n".join(lines)


def get_llm_response(api_key: str, model_name: str, context_text: str, chat_history: List[dict]) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    system_prompt = (
        "You are a soil analysis expert for paddy rice farming. "
        "Give concise, practical answers focused on what the farmer should do and why. "
        "Always ground advice in the provided sensor values and model recommendation outputs. "
        "Do NOT invent sensor values, target thresholds, or hidden agronomy rules. Include units. "
        "For each relevant output, state action + reason in one short line. "
        "If an output is 0, say 'No action needed' and explain why briefly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Use this context for the current field:\n{context_text}"},
        {
            "role": "system",
            "content": (
                "Response format: keep it concise and complete. "
                "Bullets are optional. "
                "Use this simple structure: What to do, then Why. "
                "Keep total response to 6-10 short lines."
            ),
        },
    ]
    messages.extend(chat_history)

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=320,
    )
    return completion.choices[0].message.content or "No response generated."


def require_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not found in .env")
    return api_key


def serialize_chat_messages(messages: List[ChatMessage]) -> List[dict]:
    return [message.model_dump() for message in messages]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/esp/sample")
def esp_sample_data() -> dict:
    """Sample payload format for ESP firmware testing."""
    return {
        "device_id": TEST_ESP_DEVICE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "N": TEST_PHASE1_INPUT["N"],
        "P": TEST_PHASE1_INPUT["P"],
        "K": TEST_PHASE1_INPUT["K"],
        "ph": TEST_PHASE1_INPUT["ph"],
        "EC_uS_cm": TEST_PHASE1_INPUT["EC_uS_cm"],
        "ORP_mV": TEST_PHASE2_INPUT["ORP_mV"],
    }


@app.post("/api/esp/ingest")
def esp_ingest(payload: ESPIngestRequest) -> dict:
    """ESP pushes sensor data here; server stores and auto-processes predictions."""
    try:
        phase1_block = _phase1_from_esp(payload)
        phase2_block = _phase2_from_esp(payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if phase1_block is None and phase2_block is None:
        raise HTTPException(
            status_code=400,
            detail="Payload must contain full Phase 1 sensors or ORP_mV for Phase 2.",
        )

    record = {
        "record_id": str(uuid4()),
        "device_id": payload.device_id,
        "timestamp": _make_iso_timestamp(payload.timestamp),
        "raw_sensor_data": {
            "N": payload.N,
            "P": payload.P,
            "K": payload.K,
            "ph": payload.ph,
            "EC_uS_cm": payload.EC_uS_cm,
            "ORP_mV": payload.ORP_mV,
        },
        "phase1": phase1_block,
        "phase2": phase2_block,
    }
    ESP_RECORDS.append(record)

    return {
        "status": "ingested",
        "record": record,
    }


@app.get("/api/esp/latest")
def esp_latest() -> dict:
    """Latest ESP packet with processed outputs for dashboard/mobile polling."""
    if not ESP_RECORDS:
        raise HTTPException(status_code=404, detail="No ESP data available")
    return ESP_RECORDS[-1]


@app.get("/api/esp/history")
def esp_history(limit: int = 20) -> dict:
    """Recent ESP packets for timeline/history screens."""
    safe_limit = max(1, min(limit, 200))
    data = list(ESP_RECORDS)[-safe_limit:]
    return {
        "count": len(data),
        "items": data,
    }


@app.post("/api/phase1/data")
def phase1_data(payload: Phase1Request) -> dict:
    """Return exactly the Phase 1 data shown on dashboard (no chat)."""
    sensor_input = {
        "N": payload.N,
        "P": payload.P,
        "K": payload.K,
        "ph": payload.ph,
        "EC_uS_cm": payload.EC_uS_cm,
    }

    try:
        prediction = phase1_predict(
            N=payload.N,
            P=payload.P,
            K=payload.K,
            ph=payload.ph,
            EC=payload.EC_uS_cm,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    flat_prediction = flatten_recommendations(prediction)

    return {
        "phase": "phase1",
        "sensor_data": sensor_input,
        "recommended_outputs": flat_prediction,
    }


@app.get("/api/phase1/data")
def phase1_data_get(
    N: float = TEST_PHASE1_INPUT["N"],
    P: float = TEST_PHASE1_INPUT["P"],
    K: float = TEST_PHASE1_INPUT["K"],
    ph: float = TEST_PHASE1_INPUT["ph"],
    EC_uS_cm: float = TEST_PHASE1_INPUT["EC_uS_cm"],
) -> dict:
    """Browser-friendly test route for Phase 1 predictions using query params."""
    payload = Phase1Request(N=N, P=P, K=K, ph=ph, EC_uS_cm=EC_uS_cm)
    return phase1_data(payload)


@app.post("/api/phase2/data")
def phase2_data(payload: Phase2Request) -> dict:
    """Return exactly the Phase 2 data shown on dashboard (no chat)."""
    sensor_input = {"ORP_mV": payload.ORP_mV}

    try:
        prediction = phase2_predict(ORP=payload.ORP_mV)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    flat_prediction = flatten_recommendations(prediction)

    return {
        "phase": "phase2",
        "sensor_data": sensor_input,
        "recommended_outputs": flat_prediction,
    }


@app.get("/api/phase2/data")
def phase2_data_get(
    ORP_mV: float = TEST_PHASE2_INPUT["ORP_mV"],
) -> dict:
    """Browser-friendly test route for Phase 2 predictions using a query param."""
    payload = Phase2Request(ORP_mV=ORP_mV)
    return phase2_data(payload)


@app.get("/api/test/demo")
def test_demo() -> dict:
    """Single browser-friendly route that returns the precomputed testing payloads."""
    phase1_payload = Phase1Request(**TEST_PHASE1_INPUT)
    phase2_payload = Phase2Request(**TEST_PHASE2_INPUT)
    return {
        "test_inputs": {
            "phase1": TEST_PHASE1_INPUT,
            "phase2": TEST_PHASE2_INPUT,
            "esp_device_id": TEST_ESP_DEVICE_ID,
        },
        "phase1": phase1_data(phase1_payload),
        "phase2": phase2_data(phase2_payload),
        "esp_sample": esp_sample_data(),
    }


@app.post("/api/phase1/start")
def phase1_start(payload: Phase1Request) -> dict:
    api_key = require_api_key()

    sensor_input = {
        "N": payload.N,
        "P": payload.P,
        "K": payload.K,
        "ph": payload.ph,
        "EC_uS_cm": payload.EC_uS_cm,
    }

    try:
        prediction = phase1_predict(
            N=payload.N,
            P=payload.P,
            K=payload.K,
            ph=payload.ph,
            EC=payload.EC_uS_cm,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    flat_prediction = flatten_recommendations(prediction)
    context_text = build_phase1_context(sensor_input, flat_prediction)

    first_prompt = {
        "role": "user",
        "content": (
            "Based on these Phase 1 inputs and outputs, tell the farmer what to do and why. "
            "Cover all recommendation outputs in the context, including zeros. "
            "Use only provided values."
        ),
    }
    first_response = get_llm_response(api_key, payload.model_name, context_text, [first_prompt])

    session_id = str(uuid4())
    SESSIONS[session_id] = SessionState(
        phase="phase1",
        context_text=context_text,
        model_name=payload.model_name,
        chat_messages=[
            ChatMessage(role="assistant", content=first_response),
        ],
    )

    return {
        "session_id": session_id,
        "phase": "phase1",
        "sensor_data": sensor_input,
        "recommendations": flat_prediction,
        "first_chat_response": first_response,
        "chat_messages": serialize_chat_messages(SESSIONS[session_id].chat_messages),
    }


@app.post("/api/phase2/start")
def phase2_start(payload: Phase2Request) -> dict:
    api_key = require_api_key()

    sensor_input = {"ORP_mV": payload.ORP_mV}

    try:
        prediction = phase2_predict(ORP=payload.ORP_mV)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    flat_prediction = flatten_recommendations(prediction)
    context_text = build_phase2_context(sensor_input, flat_prediction)

    first_prompt = {
        "role": "user",
        "content": (
            "Based on this Phase 2 ORP input and outputs, tell the farmer what to do and why. "
            "Cover all recommendation outputs in the context, including zeros. "
            "Use only provided values."
        ),
    }
    first_response = get_llm_response(api_key, payload.model_name, context_text, [first_prompt])

    session_id = str(uuid4())
    SESSIONS[session_id] = SessionState(
        phase="phase2",
        context_text=context_text,
        model_name=payload.model_name,
        chat_messages=[
            ChatMessage(role="assistant", content=first_response),
        ],
    )

    return {
        "session_id": session_id,
        "phase": "phase2",
        "sensor_data": sensor_input,
        "recommendations": flat_prediction,
        "first_chat_response": first_response,
        "chat_messages": serialize_chat_messages(SESSIONS[session_id].chat_messages),
    }


@app.get("/api/chat/history/{session_id}")
def chat_history(session_id: str) -> dict:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "phase": session.phase,
        "chat_messages": serialize_chat_messages(session.chat_messages),
    }


@app.post("/api/chat/followup")
def chat_followup(payload: FollowupRequest) -> dict:
    api_key = require_api_key()

    session = SESSIONS.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session.chat_messages.append(ChatMessage(role="user", content=payload.message.strip()))

    chat_history = [msg.model_dump() for msg in session.chat_messages]
    assistant_response = get_llm_response(
        api_key=api_key,
        model_name=session.model_name,
        context_text=session.context_text,
        chat_history=chat_history,
    )
    session.chat_messages.append(ChatMessage(role="assistant", content=assistant_response))

    # Keep memory bounded for long sessions.
    if len(session.chat_messages) > 30:
        session.chat_messages = session.chat_messages[-30:]

    return {
        "session_id": payload.session_id,
        "phase": session.phase,
        "assistant_response": assistant_response,
        "chat_messages": serialize_chat_messages(session.chat_messages),
    }
