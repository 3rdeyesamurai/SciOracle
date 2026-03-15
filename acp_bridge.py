import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from state_manager import SciOracleStateManager

acp_router = APIRouter()
# Binds directly to the central persistence layer
manager = SciOracleStateManager() 

@acp_router.websocket("/v1/validate")
async def acp_validate(websocket: WebSocket):
    """
    Streams code blocks or natural language "vibes" directly from IDE window 
    into the EBM validation queue.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                vibe_intent = payload.get("vibe", "")
                
                # Push the vibe directly into the Planner agent's state
                if vibe_intent:
                    manager.update_state({
                        "vibe_coding_intent": vibe_intent,
                        "validation_status": "pending",
                        "current_conjecture": None, # Force planner generation
                        "generated_code": None
                    })
                    await websocket.send_json({"status": "vibe_received", "intent": vibe_intent})
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid payload, must be JSON."})
                
    except WebSocketDisconnect:
        print("IDE Client disconnected from /acp/v1/validate")

@acp_router.websocket("/v1/state")
async def acp_state_sync(websocket: WebSocket):
    """
    Continuously syncs SciOracle's .oracle state back to the IDE, projecting energy scores, 
    graph traces, and mathematical faults inline as virtual text or diagnostic highlights.
    """
    await websocket.accept()
    try:
        last_iteration = -1
        last_status = None
        while True:
            state = manager.read_state()
            current_iteration = state.get("iteration_count", 0)
            current_status = state.get("validation_status")
            
            # Broadcast state mutation triggers back to the IDE via ACP
            if current_iteration != last_iteration or current_status != last_status:
                await websocket.send_json({
                    "acp_type": "state_diagnostic",
                    "conjecture": state.get("current_conjecture"),
                    "generated_code": state.get("generated_code"),
                    "validation_status": current_status,
                    "validation_errors": state.get("validation_errors", []),
                    "ebm_energy": state.get("ebm_energy"),
                    "is_critic_signed": state.get("critic_signature") is not None
                })
                last_iteration = current_iteration
                last_status = current_status
                
            await asyncio.sleep(0.5) # Poll rate
    except WebSocketDisconnect:
        print("IDE Client disconnected from /acp/v1/state")
