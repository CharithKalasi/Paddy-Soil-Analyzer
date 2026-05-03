# Paddy Soil Analyzer

Paddy Soil Analyzer is a two-phase soil decision support project for paddy farming.

- Phase 1 (dry soil): uses `N`, `P`, `K`, `ph`, `EC_uS_cm`
- Phase 2 (muddy soil): uses `ORP_mV`

The project includes:

- ML model training and inference
- Streamlit dashboard with phase-wise chat assistant
- FastAPI backend for Flutter/mobile integration
- ESP data ingestion endpoints for real-time sensor packets

## Project Layout

- `KrishiLink_MultiOutput_Training_Data_v4.csv` - synthetic training dataset
- `dataset_data_generator.py` - dataset generation utility
- `model_training/` - model training scripts
- `train_all_models.py` - trains all models in one run
- `predict.py` - loads saved models and returns phase predictions
- `test_orchestrator.py` - phase end-to-end sample tests
- `dashboard.py` - Streamlit UI (desktop dashboard)
- `fastapi_server.py` - API server for Flutter + ESP integration
- `Models/` - saved `.pkl` model files

## Model Breakdown

### Phase 1

- NPK model
   - Input: `N`, `P`, `K`
   - Outputs: `Urea_kg_per_acre`, `DAP_kg_per_acre`, `MOP_kg_per_acre`

- pH model
   - Inputs: `ph`, `EC_uS_cm`
   - Outputs: `Lime_kg_per_acre`, `Gypsum_kg_per_acre`

- EC model
   - Input: `EC_uS_cm`
   - Outputs: `Low_EC_Fertilizer_Boost_kg`, `Phase1_EC_Flush_Water_Liters`

### Phase 2

- ORP model
   - Input: `ORP_mV`
   - Output: `Phase2_ORP_Flood_Water_Liters`
   - Trained as binary flood decision and returned as 0 or 10000 liters

## Health Status Logic

`predict.py` returns multi-label health status based on sensor conditions and intervention needs.

It can return multiple statuses (for example nutrient stress + pH amendment needed) to avoid contradictions with recommended outputs.

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training and Test

Train all models:

```bash
python train_all_models.py
```

Run prediction sample:

```bash
python predict.py
```

Run end-to-end orchestrator tests:

```bash
python test_orchestrator.py
```

## Run Streamlit Dashboard

```bash
python -m streamlit run dashboard.py --server.port 8504
```

If a port is in use, choose another port.

## Run FastAPI Server

```bash
python -m uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

Health check:

- `GET /health`

## FastAPI Endpoints

### Dashboard-equivalent data endpoints (no chat)

- `POST /api/phase1/data`
- `POST /api/phase2/data`

Return shape:

- `phase`
- `sensor_data`
- `recommended_outputs`

### Mobile-first chat start flow

Use these when the mobile app sends sensor data to the server and needs the first recommendation plus the first LLM reply in one response:

- `POST /api/mobile/start`
- `POST /api/phase1/start`
- `POST /api/phase2/start`

Return shape:

- `session_id`
- `phase`
- `sensor_data`
- `recommended_outputs`
- `initial_chat_response`
- `chat_messages`

`initial_chat_response` and `first_chat_response` contain the same text for backward compatibility.

`/api/mobile/start` accepts:

- `phase`: `phase1` or `phase2`
- `sensor_data`: the sensor readings for that phase
- `model_name` (optional)

### Chat endpoints (Flutter can mirror dashboard chat)

- `POST /api/mobile/followup`
- `POST /api/phase1/start`
- `POST /api/phase2/start`
- `POST /api/chat/followup`
- `GET /api/chat/history/{session_id}`

Use the mobile aliases if you want the Flutter app to stay decoupled from the dashboard route naming.

For follow-up chat, send `{ session_id, message }` to `POST /api/chat/followup` and read the server reply from `reply_to_mobile` or `assistant_response`.

The server now also persists session state to disk under `server_state/chat_sessions.json`, so chat context survives a PC restart unless that file is removed.

### ESP ingestion endpoints

- `GET /api/esp/sample` - sample ESP payload format
- `POST /api/esp/phase1` - Phase 1-only ingest (`N`, `P`, `K`, `ph`, `EC_uS_cm`) with LLM recommendation
- `POST /api/esp/phase2` - Phase 2-only ingest (`ORP_mV`) with LLM recommendation
- `POST /api/esp/ingest` - backward-compatible combined ingest route
- `GET /api/esp/latest` - latest ingested + processed packet
- `GET /api/esp/history?limit=20` - recent records

ESP payload fields (send available values):

- `device_id`
- `timestamp` (optional)
- `N`, `P`, `K`, `ph`, `EC_uS_cm`, `ORP_mV`

For Flutter-first architecture, send Phase 1 and Phase 2 packets separately to:

- `POST /api/esp/phase1`
- `POST /api/esp/phase2`

Each route returns:

- `phase`
- `recommended_outputs`
- `llm_recommendation`

The mobile app can then forward the sensor data it receives from ESP to the server start endpoints above to get the recommendation and initial chat response.

`POST /api/esp/ingest` response shape:

- `recommended_outputs.phase1` (present only if full Phase 1 inputs are provided)
- `recommended_outputs.phase2` (present only if `ORP_mV` is provided)
- `llm_recommendation.phase1` / `llm_recommendation.phase2` for provided phase blocks

## Input Validation Ranges

- `N`: 0 to 100
- `P`: 0 to 70
- `K`: 0 to 55
- `ph`: 3.5 to 9.0
- `EC_uS_cm`: 0 to 3500
- `ORP_mV`: -350 to 350

Invalid inputs return HTTP 400 from API endpoints.

## Notes

- Dataset is synthetic and intentionally imbalanced.
- `Health_Status` dataset column is not used for model training.
- `Models/` directory is created automatically by training scripts.
- FastAPI in-memory session and ESP record stores reset when server restarts.