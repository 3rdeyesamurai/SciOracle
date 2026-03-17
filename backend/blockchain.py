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
