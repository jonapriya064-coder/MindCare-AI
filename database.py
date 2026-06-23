"""
MindCare AI - Database Module
==============================
Handles SQLite database initialization and helper functions.
"""

import sqlite3
import os
from config import Config


def get_db():
    """Get a database connection."""
    conn = sqlite3.connect(Config.DATABASE)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    return conn


def init_db():
    """Initialize the database with all required tables."""
    # Ensure database directory exists
    os.makedirs(os.path.dirname(Config.DATABASE), exist_ok=True)

    conn = get_db()
    cursor = conn.cursor()

    # ── Users Table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            email    TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ── Moods Table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS moods (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            emotion  TEXT    NOT NULL,
            journal  TEXT,
            date     DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── Journal Entries Table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            title    TEXT,
            text     TEXT    NOT NULL,
            date     DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # ── Quotes Table ──
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS quotes (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            quote TEXT NOT NULL,
            author TEXT
        )
    ''')

    # Insert default motivational quotes if table is empty
    cursor.execute('SELECT COUNT(*) FROM quotes')
    if cursor.fetchone()[0] == 0:
        quotes = [
            ("You don't have to control your thoughts. You just have to stop letting them control you.", "Dan Millman"),
            ("Mental health is not a destination, but a process. It's about how you drive, not where you're going.", "Noam Shpancer"),
            ("You are not alone in this. No matter how dark it feels, the sun will rise again.", "Unknown"),
            ("There is hope, even when your brain tells you there isn't.", "John Green"),
            ("Self-care is not self-indulgence. It is self-preservation.", "Audre Lorde"),
            ("Healing is not linear. Some days will be harder than others, and that's okay.", "Unknown"),
            ("You are stronger than you think, braver than you feel, and loved more than you know.", "Unknown"),
            ("It's okay to not be okay. What matters is that you keep going.", "Unknown"),
            ("Your mental health is a priority. Your happiness is an essential. Your self-care is a necessity.", "Unknown"),
            ("Be gentle with yourself. You are a child of the universe, no less than the trees and stars.", "Max Ehrmann"),
            ("Every day is a new beginning. Take a deep breath and start again.", "Unknown"),
            ("The only way out is through. One step at a time.", "Robert Frost"),
            ("You are enough, just as you are.", "Meghan Markle"),
            ("Recovery is not one and done. It is a lifelong journey that takes place one day, one step at a time.", "Unknown"),
            ("Tough times never last, but tough people do.", "Robert H. Schuller"),
            ("In the middle of difficulty lies opportunity.", "Albert Einstein"),
            ("Believe you can and you're halfway there.", "Theodore Roosevelt"),
            ("The greatest glory in living lies not in never falling, but in rising every time we fall.", "Nelson Mandela"),
            ("It does not matter how slowly you go as long as you do not stop.", "Confucius"),
            ("Start where you are. Use what you have. Do what you can.", "Arthur Ashe"),
        ]
        cursor.executemany('INSERT INTO quotes (quote, author) VALUES (?, ?)', quotes)

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")


def get_random_quote():
    """Return a random motivational quote."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT quote, author FROM quotes ORDER BY RANDOM() LIMIT 1')
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'quote': row['quote'], 'author': row['author']}
    return {'quote': 'Keep going. You are doing great!', 'author': 'MindCare AI'}
