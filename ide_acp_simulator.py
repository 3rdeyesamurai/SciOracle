import asyncio
import json
import websockets
import sys

async def listen_to_state(uri):
    """Listens to the .oracle state mutations and prints them like IDE diagnostics."""
    try:
        async with websockets.connect(uri) as websocket:
            print("[IDE State Sync] Connected to SciOracle ACP.\n> ", end="", flush=True)
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                print("\n\n" + "="*55)
                print("[ACP Diagnostic Update]")
                print(f"   Status: {data.get('validation_status').upper()}")
                
                if data.get('conjecture'):
                    print(f"   Mathematical AST Target: {data.get('conjecture')}")
                
                errs = data.get('validation_errors')
                if errs:
                    print(f"   Formal Faults Detected: {errs}")
                    
                energy = data.get('ebm_energy')
                if energy is not None:
                    print(f"   EBM Energy Score: {energy}")
                    
                if data.get('is_critic_signed'):
                    print(f"   Critic Hash Signature: Verified (Ready for Executor)")
                    
                print("="*55 + "\n> ", end="", flush=True)

    except ConnectionRefusedError:
        print("\n[Error] Could not connect to State Sync. Is the SciOracle FastAPI running?")
    except Exception as e:
        print(f"\n[State Sync Error] {e}")

async def send_vibes(uri):
    """Sends raw textual vibes from the IDE cursor payload to the Planner agent."""
    try:
        async with websockets.connect(uri) as websocket:
            print("[IDE Vibe Input] Ready. Describe your scientific intent.")
            while True:
                vibe = await asyncio.to_thread(input, "")
                if vibe.lower() in ["exit", "quit", "q"]:
                    print("Shutting down IDE Simulator.")
                    sys.exit(0)
                    
                if not vibe.strip():
                    continue
                    
                payload = {"vibe": vibe}
                await websocket.send(json.dumps(payload))
                
                response = await websocket.recv()
                ack = json.loads(response)
                print(f"   [ACP Server ACK] {ack.get('status')} - '{ack.get('intent')}'\n> ", end="", flush=True)
    except ConnectionRefusedError:
        pass # Handled by listen_to_state
    except Exception as e:
        print(f"\n[Vibe Input Error] {e}")

async def main():
    state_uri = "ws://localhost:8000/acp/v1/state"
    validate_uri = "ws://localhost:8000/acp/v1/validate"
    
    await asyncio.gather(
        listen_to_state(state_uri),
        send_vibes(validate_uri)
    )

if __name__ == "__main__":
    print("-" * 65)
    print("SciOracle IDE ACP Simulator (Proof of Concept)")
    print("This terminal simulates an IDE (VS Code/Zed) interacting with")
    print("the isolated SciOracle Symbolic engine natively over WebSockets.")
    print("-" * 65)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDisconnected.")
