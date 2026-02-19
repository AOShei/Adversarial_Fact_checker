import sqlite3
import json
from typing import List, Dict, Any

DB_NAME = "analysis_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_text TEXT,
            analysis_results TEXT,
            provider TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(report_text: str, results: List[Dict[str, Any]], provider: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Serialize results to JSON
    results_json = json.dumps(results)
    c.execute('INSERT INTO reports (report_text, analysis_results, provider) VALUES (?, ?, ?)', 
              (report_text, results_json, provider))
    conn.commit()
    conn.close()

def get_history(limit: int = 10):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('SELECT id, report_text, analysis_results, provider, timestamp FROM reports ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
