import sqlite3
import json
import cv2
from pathlib import Path
from core.base import StorageManager
import numpy as np
from typing import List, Dict, Optional

class SQLiteStorageManager(StorageManager):
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.captures_dir = Path("captures")
        self.captures_dir.mkdir(exist_ok=True)
        self.setup_database()
        
    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captures (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                image_path TEXT,
                metadata TEXT
            )
        ''')
        self.conn.commit()
    
    def save_capture(self, frame: np.ndarray, metadata: Dict) -> bool:
        try:
            image_path = self.captures_dir / f"{metadata['timestamp']}.jpg"
            cv2.imwrite(str(image_path), frame)
            
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO captures (timestamp, image_path, metadata) VALUES (?, ?, ?)",
                (metadata['timestamp'], str(image_path), json.dumps(metadata))
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False
    
    def get_captures(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM captures")
        return cursor.fetchall()
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()