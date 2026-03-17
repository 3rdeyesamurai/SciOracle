import os
import sqlite3
import time
from ebm_math_discovery import init_db, declare_theorem_if_sound
from backend.blockchain import Blockchain
from p2p_network import P2PNode

def test_minting_logic():
    db_path = "test_math_knowledge.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print("1. Initializing DB...")
    conn = init_db(db_path)
    
    # Check if blocks table exists
    cursor = conn.cursor()
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='blocks'")
    assert cursor.fetchone()[0] == 1, "Blocks table was not created"
    print("OK Blocks table exists")
    
    print("2. Minting new discovery...")
    p_nl = "Mass times acceleration equals Force"
    p_math = "m*a"
    s_nl = "Force"
    s_math = "F"
    energy = 0.01 # Below the 0.02 threshold
    is_sound = True
    
    declare_theorem_if_sound(conn, p_nl, p_math, s_nl, s_math, energy, is_sound, threshold=0.05)
    
    # Verify blocks table has the genesis and the new block
    cursor.execute("SELECT index_id, hash, energy_score FROM blocks ORDER BY index_id ASC")
    blocks = cursor.fetchall()
    
    assert len(blocks) == 2, f"Expected 2 blocks (Genesis + Discovery), found {len(blocks)}"
    assert blocks[0][0] == 0, "First block should be Genesis (index 0)"
    assert blocks[1][0] == 1, "Second block should be index 1"
    assert blocks[1][2] == 0.01, "Energy score mismatch"
    print("OK Discovery successfully minted into local blockchain DB")
    
    print("3. Testing P2P Node DB Sync...")
    node = P2PNode(host='127.0.0.1', port=5001, db_path=db_path)
    assert len(node.blockchain.chain) == 2, "P2PNode failed to load chain from DB"
    assert node.blockchain.chain[-1].energy_score == 0.01, "Loaded chain data mismatch"
    print("OK P2PNode successfully synced chain from DB")
    
    print("\nAll Blockchain and Minting tests passed successfully! 🚀")
    
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    try:
        test_minting_logic()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCaught exception: {e}")
