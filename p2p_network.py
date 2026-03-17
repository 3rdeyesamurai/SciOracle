import asyncio
import json
import sqlite3
import time
from typing import Set, Dict, Any
from backend.blockchain import Block, Blockchain

class P2PNode:
    def __init__(self, host: str = '0.0.0.0', port: int = 5000, db_path: str = "math_knowledge.db"):
        self.host = host
        self.port = port
        self.db_path = db_path
        self.peers: Set[str] = set()
        self.blockchain = Blockchain()
        self._sync_chain_from_db()

    def _get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def _sync_chain_from_db(self):
        """Loads the chain state from the SQLite database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='blocks'")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return

        cursor.execute("SELECT index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address, merkle_root FROM blocks ORDER BY index_id ASC")
        rows = cursor.fetchall()
        
        if rows:
            chain = []
            for row in rows:
                block_data = {
                    "index": row[0],
                    "timestamp": row[1],
                    "data": json.loads(row[2]),
                    "prev_hash": row[3],
                    "hash": row[4],
                    "energy_score": row[5],
                    "miner_address": row[6],
                    "merkle_root": row[7]
                }
                chain.append(Block.from_dict(block_data))
            
            if self.blockchain.is_valid_chain(chain):
                self.blockchain.chain = chain
            else:
                print(f"[DEBUG] _sync_chain_from_db: is_valid_chain returned False for chain of length {len(chain)}")
        else:
            print("[DEBUG] _sync_chain_from_db: No rows found in blocks table")
        conn.close()

    def _save_block_to_db(self, block: Block):
        """Persist a single verified block to the DB."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO blocks (index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address, merkle_root) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (block.index, block.timestamp, json.dumps(block.data), block.prev_hash, block.hash, block.energy_score, block.miner_address, block.merkle_root)
        )
        conn.commit()
        conn.close()

    async def start_server(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f"P2P Network Node starting on {self.host}:{self.port}")
        
        # Start the background task to poll the DB for new locally mined blocks
        asyncio.create_task(self.poll_local_db())
        
        async with server:
            await server.serve_forever()

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                message = json.loads(data.decode())
                await self.process_message(message, writer, addr)
        except Exception as e:
            print(f"Connection error with {addr}: {e}")
        finally:
            writer.close()

    async def process_message(self, message: Dict[str, Any], writer, addr):
        msg_type = message.get("type")
        
        if msg_type == "HELLO":
            peer_address = f"{addr[0]}:{message.get('port', 5000)}"
            self.peers.add(peer_address)
            
            # Send current chain length
            response = {"type": "STATUS", "chain_length": len(self.blockchain.chain)}
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
            
        elif msg_type == "NEW_BLOCK":
            block_data = message.get("block")
            if block_data:
                new_block = Block.from_dict(block_data)
                self._sync_chain_from_db() # Ensure we have latest local state
                
                if self.blockchain.add_block(new_block):
                    print(f"[P2P] Received and validated valid NEW_BLOCK: {new_block.index} from network.")
                    self._save_block_to_db(new_block)
                else:
                    # Might be a fork or we are behind, request full chain sync
                    if new_block.index > len(self.blockchain.chain):
                        req = {"type": "GET_CHAIN"}
                        writer.write((json.dumps(req) + "\n").encode())
                        await writer.drain()

        elif msg_type == "GET_CHAIN":
            self._sync_chain_from_db()
            response = {"type": "RESP_CHAIN", "chain": self.blockchain.to_list()}
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
            
        elif msg_type == "RESP_CHAIN":
            chain_data = message.get("chain", [])
            self._sync_chain_from_db() # Sync current DB first
            if self.blockchain.replace_chain(chain_data):
                print(f"[P2P] Successfully synced chain from peer. New length: {len(self.blockchain.chain)}")
                # Replace local DB DB entries
                conn = self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM blocks")
                
                for block in self.blockchain.chain:
                    cursor.execute(
                        "INSERT INTO blocks (index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address, merkle_root) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (block.index, block.timestamp, json.dumps(block.data), block.prev_hash, block.hash, block.energy_score, block.miner_address, block.merkle_root)
                    )
                conn.commit()
                conn.close()
                
    async def connect_to_peer(self, host: str, port: int):
        try:
            reader, writer = await asyncio.open_connection(host, port)
            hello_msg = {"type": "HELLO", "port": self.port}
            writer.write((json.dumps(hello_msg) + "\n").encode())
            await writer.drain()
            
            # Start background reader for this connection
            asyncio.create_task(self.handle_client(reader, writer))
            self.peers.add(f"{host}:{port}")
            print(f"[P2P] Connected to peer {host}:{port}")
            return writer
        except Exception as e:
            print(f"[P2P] Failed to connect to {host}:{port} - {e}")
            return None

    async def broadcast_message(self, message: Dict[str, Any]):
        disconnected_peers = set()
        for peer in self.peers:
            host, port_str = peer.split(":")
            port = int(port_str)
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.write((json.dumps(message) + "\n").encode())
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            except Exception:
                disconnected_peers.add(peer)
        
        self.peers -= disconnected_peers

    async def poll_local_db(self):
        """Continuously check the DB for newly minted local blocks to broadcast."""
        while True:
            await asyncio.sleep(5)
            # Fetch local latest block directly
            conn = self._get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='blocks'")
                if cursor.fetchone()[0] == 0:
                    conn.close()
                    continue

                cursor.execute("SELECT index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address, merkle_root FROM blocks ORDER BY index_id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    latest_index = row[0]
                    # Check if our in-memory chain is behind the DB
                    if not self.blockchain.chain or latest_index > self.blockchain.get_latest_block().index:
                        block_data = {
                            "index": row[0],
                            "timestamp": row[1],
                            "data": json.loads(row[2]),
                            "prev_hash": row[3],
                            "hash": row[4],
                            "energy_score": row[5],
                            "miner_address": row[6],
                            "merkle_root": row[7]
                        }
                        new_block = Block.from_dict(block_data)
                        
                        # Sync completely to be safe
                        self._sync_chain_from_db()
                        
                        # Broadcast this newly found block
                        print(f"[P2P] DB Poll detected new local block {new_block.index}. Broadcasting...")
                        msg = {"type": "NEW_BLOCK", "block": new_block.to_dict()}
                        await self.broadcast_message(msg)
            except Exception as e:
                print(f"[P2P] Error polling local DB: {e}")
            finally:
                conn.close()

def run_p2p_node(host='0.0.0.0', port=5000, initial_peers=None):
    # Setup event loop for this process
    node = P2PNode(host=host, port=port)
    loop = asyncio.get_event_loop()
    
    if initial_peers:
        for p in initial_peers:
            p_host, p_port = p.split(":")
            loop.run_until_complete(node.connect_to_peer(p_host, int(p_port)))
            
    try:
        loop.run_until_complete(node.start_server())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    peer = sys.argv[2] if len(sys.argv) > 2 else None
    
    peers = [peer] if peer else []
    run_p2p_node(port=port, initial_peers=peers)
