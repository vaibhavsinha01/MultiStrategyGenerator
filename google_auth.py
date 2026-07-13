"""Google OAuth helpers — reads credentials from google_oauth.json or env."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

ROOT = Path(__file__).resolve().parent

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = "openid email profile"


def _load_oauth_config() -> dict[str, str]:
    path = os.environ.get("GOOGLE_OAUTH_JSON_PATH", str(ROOT / "google_oauth.json"))
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        web = data.get("web", data)
        return {
            "client_id": web.get("client_id", ""),
            "client_secret": web.get("client_secret", ""),
        }
    return {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
    }


def oauth_redirect_uri(request_base: str) -> str:
    explicit = os.environ.get("GOOGLE_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    return f"{request_base.rstrip('/')}/auth/google/callback"


def build_google_auth_url(request_base: str, state: str) -> str:
    cfg = _load_oauth_config()
    if not cfg["client_id"]:
        raise RuntimeError("Google OAuth client_id is not configured.")
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": oauth_redirect_uri(request_base),
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "online",
        "include_granted_scopes": "true",
        "state": state,
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_user(code: str, request_base: str) -> dict[str, Any]:
    cfg = _load_oauth_config()
    redirect_uri = oauth_redirect_uri(request_base)
    token_resp = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json().get("access_token")
    if not access_token:
        raise RuntimeError("Google token exchange failed.")

    user_resp = requests.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    user_resp.raise_for_status()
    profile = user_resp.json()
    if not profile.get("email"):
        raise RuntimeError("Google profile did not include an email.")
    return profile


def new_oauth_state() -> str:
    return secrets.token_urlsafe(24)
