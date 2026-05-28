from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
import psycopg2.extras
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

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# def _connect():
#     return psycopg2.connect(DATABASE_URL)
def _connect():
    print(f"Connecting to DB: {DATABASE_URL[:60]}...")
    return psycopg2.connect(DATABASE_URL, sslmode="require")

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
            """
        )


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
            INSERT INTO users (email, full_name, hashed_password)
            VALUES (%s, %s, %s)
            RETURNING id, email, full_name, created_at
            """,
            (
                email.strip().lower(),
                full_name.strip(),
                hashed,
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
