"""PayPal checkout integration."""

from __future__ import annotations

import base64
import os
from typing import Any

import requests

PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.environ.get("PAYPAL_CLIENT_SECRET", "")
PAYPAL_MODE = os.environ.get("PAYPAL_MODE", "sandbox").lower()
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:8031")
PRO_PLAN_PRICE = os.environ.get("PAYPAL_PLAN_PRICE", "49.00")
PRO_PLAN_CURRENCY = os.environ.get("PAYPAL_PLAN_CURRENCY", "USD")


def _api_base() -> str:
    return (
        "https://api-m.sandbox.paypal.com"
        if PAYPAL_MODE == "sandbox"
        else "https://api-m.paypal.com"
    )


def _access_token() -> str:
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise RuntimeError("PayPal credentials are not configured in .env")
    auth = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        f"{_api_base()}/v1/oauth2/token",
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=20,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise RuntimeError("PayPal access token missing.")
    return token


def create_checkout_order(user_id: int) -> dict[str, Any]:
    token = _access_token()
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "reference_id": f"user-{user_id}",
                "description": "Simurgh Trading Pro — strategy exports & premium dashboard",
                "amount": {
                    "currency_code": PRO_PLAN_CURRENCY,
                    "value": PRO_PLAN_PRICE,
                },
            }
        ],
        "application_context": {
            "brand_name": "Simurgh Trading",
            "landing_page": "NO_PREFERENCE",
            "user_action": "PAY_NOW",
            "return_url": f"{APP_BASE_URL.rstrip('/')}/payment/success",
            "cancel_url": f"{APP_BASE_URL.rstrip('/')}/payment/cancel",
        },
    }
    resp = requests.post(
        f"{_api_base()}/v2/checkout/orders",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def capture_order(order_id: str) -> dict[str, Any]:
    token = _access_token()
    resp = requests.post(
        f"{_api_base()}/v2/checkout/orders/{order_id}/capture",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()
