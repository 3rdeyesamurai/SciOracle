import hashlib
import time
import json
from typing import List, Dict, Any, Optional

class Block:
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], prev_hash: str, energy_score: float, miner_address: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data # This contains the "Natural Law" (e.g. signature, problem_math, solution_math)
        self.prev_hash = prev_hash
        self.energy_score = energy_score
        self.miner_address = miner_address
        self.merkle_root = self._calculate_merkle_root()
        self.hash = self.calculate_hash()

    def _calculate_merkle_root(self) -> str:
        # For simplicity, we just hash the data payload for the single discovery in this block.
        # In a real blockchain with multiple transactions per block, this would be a full Merkle Tree.
        data_string = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

    def calculate_hash(self) -> str:
        block_string = f"{self.index}{self.timestamp}{self.merkle_root}{self.prev_hash}{self.energy_score}{self.miner_address}"
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
            "energy_score": self.energy_score,
            "miner_address": self.miner_address,
            "merkle_root": self.merkle_root
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        block = cls(
            index=data['index'],
            timestamp=data['timestamp'],
            data=data['data'],
            prev_hash=data['prev_hash'],
            energy_score=data['energy_score'],
            miner_address=data['miner_address']
        )
        block.hash = data['hash']
        block.merkle_root = data.get('merkle_root', block._calculate_merkle_root())
        return block

class Blockchain:
    def __init__(self, difficulty_threshold: float = 0.02):
        self.chain: List[Block] = []
        self.difficulty_threshold = difficulty_threshold
        self.create_genesis_block()

    def create_genesis_block(self):
        if not self.chain:
            genesis_data = {"message": "Genesis Block - The Beginning of SciOracle PoD Cosmos"}
            genesis_block = Block(0, 1700000000.0, genesis_data, "0" * 64, 0.0, "Genesis")
            self.chain.append(genesis_block)

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, new_block: Block) -> bool:
        if self.is_valid_new_block(new_block, self.get_latest_block()):
            self.chain.append(new_block)
            return True
        return False

    def is_valid_new_block(self, new_block: Block, prev_block: Block) -> bool:
        if prev_block.index + 1 != new_block.index:
            return False
        if prev_block.hash != new_block.prev_hash:
            return False
        if new_block.calculate_hash() != new_block.hash:
            return False
        if new_block.energy_score > self.difficulty_threshold and new_block.index > 0:
            return False
        return True

    def replace_chain(self, new_chain_data: List[Dict[str, Any]]) -> bool:
        new_chain = [Block.from_dict(b) for b in new_chain_data]
        if self.is_valid_chain(new_chain) and len(new_chain) > len(self.chain):
            self.chain = new_chain
            return True
        return False

    def is_valid_chain(self, chain_to_test: List[Block]) -> bool:
        if chain_to_test[0].hash != self.chain[0].hash:
            return False
        
        for i in range(1, len(chain_to_test)):
            current_block = chain_to_test[i]
            prev_block = chain_to_test[i - 1]
            if not self.is_valid_new_block(current_block, prev_block):
                return False
        return True

    def to_list(self) -> List[Dict[str, Any]]:
        return [block.to_dict() for block in self.chain]

class DiscoveryLedger:
    def __init__(self, db_conn, initial_difficulty=0.02, blocks_per_adjustment=10, expected_block_time=10.0):
        self.conn = db_conn
        self.initial_difficulty = initial_difficulty
        self.blocks_per_adjustment = blocks_per_adjustment
        self.expected_block_time = expected_block_time

    def get_current_difficulty(self) -> float:
        cursor = self.conn.cursor()
        cursor.execute("SELECT timestamp, energy_score FROM blocks ORDER BY index_id DESC LIMIT ?", (self.blocks_per_adjustment,))
        rows = cursor.fetchall()
        
        if len(rows) < self.blocks_per_adjustment:
            return self.initial_difficulty
            
        # Reverse rows to chronological order
        rows.reverse()
        
        time_taken = rows[-1][0] - rows[0][0]
        expected_time = self.expected_block_time * self.blocks_per_adjustment
        
        # We look at the average difficulty in this window
        avg_difficulty = sum(r[1] for r in rows) / len(rows)

        # Base difficulty could just be adjusted by time ratio. 
        # Lower energy_score means harder.
        # If time_taken < expected_time, blocks were found too fast, so energy threshold must DECREASE (harder).
        
        # Prevent division by zero and absurd times
        time_taken = max(1.0, time_taken)
        ratio = time_taken / expected_time
        
        # Clamp adjustment to handle extreme spikes
        ratio = max(0.5, min(2.0, ratio))
        
        new_difficulty = avg_difficulty * ratio
        # Do not allow difficulty to become trivially easy (e.g. > 0.1) or impossibly hard (e.g. < 0.0001)
        new_difficulty = max(0.0001, min(0.1, new_difficulty))
        
        return new_difficulty

    def mint_block(self, energy_value: float, is_sound: bool, message: Dict[str, Any], miner_address: str = "SciOracle_Local_Miner"):
        if not is_sound:
            return False

        current_diff = self.get_current_difficulty()
        if energy_value > current_diff:
            return False

        cursor = self.conn.cursor()
        cursor.execute("SELECT hash, index_id FROM blocks ORDER BY index_id DESC LIMIT 1")
        last_block_row = cursor.fetchone()

        if last_block_row:
            prev_hash = last_block_row[0]
            new_index = last_block_row[1] + 1
        else:
            genesis = Block(0, 1700000000.0, {"message": "Genesis Block - The Beginning of SciOracle PoD Cosmos"}, "0"*64, 0.0, "Genesis")
            cursor.execute(
                "INSERT INTO blocks (index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address, merkle_root) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (genesis.index, genesis.timestamp, json.dumps(genesis.data), genesis.prev_hash, genesis.hash, genesis.energy_score, genesis.miner_address, genesis.merkle_root)
            )
            prev_hash = genesis.hash
            new_index = 1
            
        block = Block(
            index=new_index,
            timestamp=time.time(),
            data=message,
            prev_hash=prev_hash,
            energy_score=energy_value,
            miner_address=miner_address
        )
        
        cursor.execute(
            "INSERT INTO blocks (index_id, timestamp, data_json, prev_hash, hash, energy_score, miner_address, merkle_root) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (block.index, block.timestamp, json.dumps(block.data), block.prev_hash, block.hash, block.energy_score, block.miner_address, block.merkle_root)
        )
        self.conn.commit()
        return True

