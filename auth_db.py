
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
import psycopg2.extras
from urllib.parse import urlparse
from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv


load_dotenv()


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:5432/quant_db",
)
SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-local-dev-secret")
ALGORITHM = os.environ.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# ── credit system config ──────────────────────────────────────────────────────
USER_CREDITS = int(os.environ.get("USER_CREDITS", "10"))
PREMIUM_CREDITS = int(os.environ.get("PREMIUM_CREDITS", "100"))
ADMIN_CREDITS = int(os.environ.get("ADMIN_CREDITS", "100000"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# def _connect():
#     return psycopg2.connect(DATABASE_URL)
def _connect():
    """
    Connect to Postgres.

    - Localhost typically does NOT run with SSL, so forcing sslmode=require breaks local dev.
    - Hosted providers often require SSL.
    """
    print(f"Connecting to DB: {DATABASE_URL[:60]}...")

    sslmode = os.environ.get("DATABASE_SSLMODE", "").strip().lower()
    if not sslmode:
        try:
            parsed = urlparse(DATABASE_URL)
            host = (parsed.hostname or "").lower()
            if host in {"localhost", "127.0.0.1", "::1"}:
                sslmode = "disable"
            else:
                sslmode = "require"
        except Exception:
            sslmode = "require"

    return psycopg2.connect(DATABASE_URL, sslmode=sslmode)

def init_db() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                hashed_password TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS app_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                buy_probability DOUBLE PRECISION NOT NULL,
                sell_probability DOUBLE PRECISION NOT NULL,
                model_version TEXT NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                provider TEXT NOT NULL DEFAULT 'paypal',
                status TEXT NOT NULL,
                order_id TEXT,
                amount NUMERIC,
                currency TEXT DEFAULT 'USD',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS credit_ledger (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                delta INTEGER NOT NULL,
                reason TEXT NOT NULL,
                balance_after INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_premium BOOLEAN NOT NULL DEFAULT FALSE")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id TEXT")
        cur.execute(
            f"ALTER TABLE users ADD COLUMN IF NOT EXISTS credits INTEGER NOT NULL DEFAULT {USER_CREDITS}"
        )
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_google_id_idx ON users(google_id) WHERE google_id IS NOT NULL")


def user_has_premium(user: dict[str, Any]) -> bool:
    return bool(user.get("is_admin") or user.get("is_premium"))


def set_user_premium(user_id: int, premium: bool = True) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE users SET is_premium=%s WHERE id=%s", (premium, user_id))
        if premium:
            # Bump the user up to the premium credit allotment, but never lower
            # a balance they already have (e.g. an admin, or someone who
            # already has more credits than the premium grant).
            cur.execute(
                "UPDATE users SET credits = GREATEST(credits, %s) WHERE id=%s",
                (PREMIUM_CREDITS, user_id),
            )


def record_subscription(user_id: int, order_id: str, status: str, amount: float, currency: str = "USD") -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO subscriptions (user_id, provider, status, order_id, amount, currency)
            VALUES (%s, 'paypal', %s, %s, %s, %s)
            """,
            (user_id, status, order_id, amount, currency),
        )


# ── credit helpers ────────────────────────────────────────────────────────────

def get_user_credits(user: dict[str, Any]) -> int:
    """Read credits off an already-fetched user dict (may be stale if the
    balance changed elsewhere in the same request; use get_user_by_id for a
    fresh read when that matters)."""
    return int(user.get("credits", 0) or 0)


def user_has_credits(user: dict[str, Any], amount: int) -> bool:
    return get_user_credits(user) >= amount


def deduct_credits(user_id: int, amount: int, reason: str = "") -> int:
    """
    Atomically deduct `amount` credits from user_id, but only if they have
    enough. Returns the new balance. Raises ValueError if insufficient.
    """
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            UPDATE users
            SET credits = credits - %s
            WHERE id = %s AND credits >= %s
            RETURNING credits
            """,
            (amount, user_id, amount),
        )
        row = cur.fetchone()
        if row is None:
            # Either the user doesn't exist, or didn't have enough credits.
            cur2 = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur2.execute("SELECT credits FROM users WHERE id=%s", (user_id,))
            current = cur2.fetchone()
            current_balance = int(current["credits"]) if current else 0
            raise ValueError(
                f"Insufficient credits: have {current_balance}, need {amount}"
            )
        new_balance = int(row["credits"])
        cur.execute(
            "INSERT INTO credit_ledger (user_id, delta, reason, balance_after) VALUES (%s, %s, %s, %s)",
            (user_id, -amount, reason, new_balance),
        )
        return new_balance


def add_credits(user_id: int, amount: int, reason: str = "") -> int:
    """Grant credits (e.g. top-ups, admin adjustments). Returns new balance."""
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "UPDATE users SET credits = credits + %s WHERE id=%s RETURNING credits",
            (amount, user_id),
        )
        row = cur.fetchone()
        new_balance = int(row["credits"]) if row else 0
        cur.execute(
            "INSERT INTO credit_ledger (user_id, delta, reason, balance_after) VALUES (%s, %s, %s, %s)",
            (user_id, amount, reason, new_balance),
        )
        return new_balance


def set_user_credits(user_id: int, amount: int, reason: str = "admin_set") -> int:
    """Set a user's credit balance to an exact value."""
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "UPDATE users SET credits=%s WHERE id=%s RETURNING credits",
            (amount, user_id),
        )
        row = cur.fetchone()
        new_balance = int(row["credits"]) if row else 0
        cur.execute(
            "INSERT INTO credit_ledger (user_id, delta, reason, balance_after) VALUES (%s, %s, %s, %s)",
            (user_id, amount, reason, new_balance),
        )
        return new_balance


def ensure_admin_user() -> None:
    email = os.environ.get("ADMIN_EMAIL", "vaibhavrajsinha099@gmail.com").strip().lower()
    password = os.environ.get("ADMIN_PASSWORD", "Hello#2004")
    full_name = os.environ.get("ADMIN_NAME", "Vaibhav Raj Sinha")
    existing = get_user_by_email(email)
    if existing:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET is_admin=TRUE, is_premium=TRUE, full_name=%s, hashed_password=%s,
                    credits = GREATEST(credits, %s)
                WHERE id=%s
                """,
                (full_name, hash_password(password), ADMIN_CREDITS, existing["id"]),
            )
        return
    user = create_user(email, full_name, password)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE users SET is_admin=TRUE, is_premium=TRUE, credits=%s WHERE id=%s",
            (ADMIN_CREDITS, user["id"]),
        )


def create_or_get_google_user(email: str, full_name: str, google_id: str) -> dict[str, Any]:
    email = email.strip().lower()
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE google_id=%s OR lower(email)=lower(%s) LIMIT 1", (google_id, email))
        row = cur.fetchone()
        if row:
            user = dict(row)
            cur.execute(
                "UPDATE users SET google_id=%s, full_name=%s WHERE id=%s",
                (google_id, full_name.strip() or user.get("full_name", ""), user["id"]),
            )
            cur.execute("SELECT * FROM users WHERE id=%s", (user["id"],))
            return dict(cur.fetchone())
        placeholder_pw = hash_password(os.urandom(24).hex())
        cur.execute(
            """
            INSERT INTO users (email, full_name, hashed_password, google_id, credits)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, email, full_name, is_admin, is_premium, google_id, credits, created_at
            """,
            (email, full_name.strip() or email.split("@")[0], placeholder_pw, google_id, USER_CREDITS),
        )
        return dict(cur.fetchone())


# def hash_password(password: str) -> str:
#     return pwd_context.hash(password)

# def hash_password(password: str) -> str:
#     try:
#         hashed = pwd_context.hash(password)
#         print("PASSWORD HASH SUCCESS")
#         return hashed
#     except Exception as e:
#         print(f"HASH PASSWORD ERROR: {repr(e)}")
#         raise

def hash_password(password: str) -> str:

    password = password.strip()

    print(f"PASSWORD LENGTH: {len(password)}")
    print(f"PASSWORD BYTES: {len(password.encode('utf-8'))}")

    try:
        hashed = pwd_context.hash(password)
        print("PASSWORD HASH SUCCESS")
        return hashed

    except Exception as e:
        print(f"HASH PASSWORD ERROR: {repr(e)}")
        raise

# def verify_password(password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(password, hashed_password)

def verify_password(password: str, hashed_password: str) -> bool:
    try:
        # result = pwd_context.verify(password, hashed_password)
        result = pwd_context.verify(password.strip(), hashed_password)
        print(f"VERIFY PASSWORD RESULT: {result}")
        return result
    except Exception as e:
        print(f"VERIFY PASSWORD ERROR: {repr(e)}")
        raise

def get_user_by_email(email: str) -> dict[str, Any] | None:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE lower(email)=lower(%s)", (email.strip(),))
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# def create_user(email: str, full_name: str, password: str) -> dict[str, Any]:
#     with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
#         cur.execute(
#             """
#             INSERT INTO users (email, full_name, hashed_password)
#             VALUES (%s, %s, %s)
#             RETURNING id, email, full_name, created_at
#             """,
#             (email.strip().lower(), full_name.strip(), hash_password(password)),
#         )
#         return dict(cur.fetchone())
def create_user(email: str, full_name: str, password: str) -> dict[str, Any]:

    print(f"Creating user: {email}")

    with _connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

        hashed = hash_password(password)

        print(f"HASH GENERATED: {hashed[:20]}")

        cur.execute(
            """
            INSERT INTO users (email, full_name, hashed_password, credits)
            VALUES (%s, %s, %s, %s)
            RETURNING id, email, full_name, credits, created_at
            """,
            (
                email.strip().lower(),
                full_name.strip(),
                hashed,
                USER_CREDITS,
            ),
        )

        user = dict(cur.fetchone())

        print(f"USER INSERTED: {user}")

        return user

def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict[str, Any]) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {**data, "exp": expires}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.cookies.get("access_token")


def get_current_user(request: Request) -> dict[str, Any]:
    token = _token_from_request(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def optional_user(request: Request) -> dict[str, Any] | None:
    try:
        return get_current_user(request)
    except HTTPException:
        return None


def log_event(event_type: str, message: str, user_id: int | None = None, metadata: dict | None = None) -> None:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app_logs (user_id, event_type, message, metadata) VALUES (%s, %s, %s, %s)",
                (user_id, event_type, message, psycopg2.extras.Json(metadata or {})),
            )
    except Exception:
        pass


def save_prediction(
    symbol: str,
    timeframe: str,
    buy_probability: float,
    sell_probability: float,
    model_version: str,
    payload: dict,
    user_id: int | None = None,
) -> None:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                (user_id, symbol, timeframe, buy_probability, sell_probability, model_version, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    symbol.lower(),
                    timeframe.lower(),
                    buy_probability,
                    sell_probability,
                    model_version,
                    psycopg2.extras.Json(payload),
                ),
            )
    except Exception:
        pass