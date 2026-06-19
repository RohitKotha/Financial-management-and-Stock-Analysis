"""
Database utilities for storing and retrieving financial data
"""

import hashlib
import os
import re
import sqlite3
import pandas as pd
from . import config


class Database:
    def __init__(self, db_path=config.DATABASE_PATH):
        self.db_path = db_path
        self.current_user_id = None
        self.init_database()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def set_current_user(self, user_id):
        self.current_user_id = user_id

    def clear_current_user(self):
        self.current_user_id = None

    def _get_effective_user_id(self, user_id=None):
        return user_id if user_id is not None else self.current_user_id

    def _require_user_id(self, user_id=None):
        effective_user_id = self._get_effective_user_id(user_id)
        if effective_user_id is None:
            raise ValueError("User not authenticated")
        return effective_user_id

    def _table_columns(self, conn, table_name):
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cursor.fetchall()]

    def _ensure_user_column(self, conn, table_name):
        columns = self._table_columns(conn, table_name)
        if 'user_id' not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN user_id INTEGER")

    def _ensure_users_email_column(self, conn):
        columns = self._table_columns(conn, 'users')
        if 'email' not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")

    def _migrate_budgets_table(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='budgets'")
        row = cursor.fetchone()

        if row is None:
            conn.execute('''
                CREATE TABLE budgets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL,
                    month TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, category, month),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            return

        existing_sql = row[0] or ""
        if "UNIQUE(user_id, category, month)" in existing_sql:
            columns = self._table_columns(conn, 'budgets')
            if 'user_id' in columns:
                return

        columns = self._table_columns(conn, 'budgets')
        has_user_id = 'user_id' in columns

        conn.execute('''
            CREATE TABLE IF NOT EXISTS budgets_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                month TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category, month),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        if has_user_id:
            conn.execute('''
                INSERT INTO budgets_new (id, user_id, category, amount, month, created_at)
                SELECT id, user_id, category, amount, month, created_at FROM budgets
            ''')
        else:
            conn.execute('''
                INSERT INTO budgets_new (id, user_id, category, amount, month, created_at)
                SELECT id, NULL, category, amount, month, created_at FROM budgets
            ''')

        conn.execute("DROP TABLE budgets")
        conn.execute("ALTER TABLE budgets_new RENAME TO budgets")

    def init_database(self):
        """Initialize database tables and migrate legacy schema."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self._ensure_users_email_column(conn)

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                purchase_price REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                goal_name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL DEFAULT 0,
                target_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        self._ensure_user_column(conn, 'transactions')
        self._ensure_user_column(conn, 'portfolio')
        self._ensure_user_column(conn, 'financial_goals')
        self._migrate_budgets_table(conn)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_budgets_user ON budgets(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_user ON financial_goals(user_id)")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email COLLATE NOCASE) WHERE email IS NOT NULL")

        conn.commit()
        conn.close()

    def _hash_password(self, password):
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 120000)
        return f"{salt.hex()}${key.hex()}"

    def _verify_password(self, password, stored_hash):
        try:
            salt_hex, key_hex = stored_hash.split('$')
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(key_hex)
            actual = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 120000)
            return actual == expected
        except Exception:
            return False

    def create_user(self, username, password, email=None):
        username = (username or '').strip()
        email = (email or '').strip().lower()
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        if not email:
            return False, "Email is required"
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            return False, "Please enter a valid email address"
        if len(password or '') < 6:
            return False, "Password must be at least 6 characters"

        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, self._hash_password(password))
            )
            conn.commit()
            return True, "Account created successfully"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists"
        finally:
            conn.close()

    def authenticate_user(self, username, password):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ? COLLATE NOCASE",
            ((username or '').strip(),)
        )
        user = cursor.fetchone()
        conn.close()

        if not user:
            return False, "Invalid username or password", None

        if not self._verify_password(password or '', user[2]):
            return False, "Invalid username or password", None

        return True, "Login successful", {'id': user[0], 'username': user[1]}

    def add_transaction(self, date, description, amount, category, trans_type, user_id=None):
        """Add a new transaction"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (user_id, date, description, amount, category, type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (effective_user_id, date, description, amount, category, trans_type))
        conn.commit()
        conn.close()

    def get_transactions(self, start_date=None, end_date=None, trans_type=None, user_id=None):
        """Retrieve transactions with optional filters"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        query = "SELECT * FROM transactions WHERE user_id = ?"
        params = [effective_user_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if trans_type:
            query += " AND type = ?"
            params.append(trans_type)

        query += " ORDER BY date DESC"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def set_budget(self, category, amount, month, user_id=None):
        """Set or update budget for a category"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO budgets (user_id, category, amount, month)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, category, month)
            DO UPDATE SET amount = excluded.amount, created_at = CURRENT_TIMESTAMP
        ''', (effective_user_id, category, amount, month))
        conn.commit()
        conn.close()

    def get_budgets(self, month=None, user_id=None):
        """Get budgets for a specific month"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        if month:
            df = pd.read_sql_query(
                "SELECT * FROM budgets WHERE user_id = ? AND month = ?",
                conn,
                params=[effective_user_id, month]
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM budgets WHERE user_id = ?",
                conn,
                params=[effective_user_id]
            )
        conn.close()
        return df

    def add_portfolio_item(self, symbol, quantity, purchase_price, purchase_date, user_id=None):
        """Add a stock to portfolio"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO portfolio (user_id, symbol, quantity, purchase_price, purchase_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (effective_user_id, symbol, quantity, purchase_price, purchase_date))
        conn.commit()
        conn.close()

    def get_portfolio(self, user_id=None):
        """Get current portfolio"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM portfolio WHERE user_id = ?",
            conn,
            params=[effective_user_id]
        )
        conn.close()
        return df

    def add_financial_goal(self, goal_name, target_amount, target_date, current_amount=0, user_id=None):
        """Add a financial goal"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO financial_goals (user_id, goal_name, target_amount, current_amount, target_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (effective_user_id, goal_name, target_amount, current_amount, target_date))
        conn.commit()
        conn.close()

    def get_financial_goals(self, user_id=None):
        """Get all financial goals"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM financial_goals WHERE user_id = ?",
            conn,
            params=[effective_user_id]
        )
        conn.close()
        return df

    def update_goal_progress(self, goal_id, current_amount, user_id=None):
        """Update progress on a financial goal"""
        effective_user_id = self._require_user_id(user_id)
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE financial_goals SET current_amount = ?
            WHERE id = ? AND user_id = ?
        ''', (current_amount, goal_id, effective_user_id))
        conn.commit()
        conn.close()
