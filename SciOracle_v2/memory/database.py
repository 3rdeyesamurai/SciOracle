import sqlite3
import json

class TheoremDatabase:
    """
    SQLite-backed storage system. Stores verified mathematical identities, 
    proof objects and network meta-data for future training passes.
    """
    def __init__(self, db_path: str = "memory/theorem.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create base schema if missing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS theorems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                initial_equation TEXT NOT NULL,
                final_equation TEXT NOT NULL,
                proof_json TEXT NOT NULL,
                complexity INTEGER DEFAULT 1,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(initial_equation, final_equation)
            )
            ''')
            conn.commit()

    def insert_theorem(self, initial: str, final: str, proof_dict: dict, complexity: int = 1):
        """
        Record a novel theorem proof.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT OR IGNORE INTO theorems 
                       (initial_equation, final_equation, proof_json, complexity) 
                       VALUES (?, ?, ?, ?)''',
                    (initial, final, json.dumps(proof_dict), complexity)
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Database error on insert: {e}")

    def get_all_theorems(self):
        """Fetch all discovered theorems."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, initial_equation, final_equation, complexity FROM theorems")
            return cursor.fetchall()
