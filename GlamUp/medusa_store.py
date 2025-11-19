import os
import requests
from typing import Dict, Any


BASE_URL = os.environ.get("MEDUSA_BASE_URL", "http://13.200.100.177:9000")
PUBLISHABLE_KEY = os.environ.get("MEDUSA_PUBLISHABLE_KEY")


def list_store_products(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """
    Fetch products from the Medusa /store/products API.

    Requirements:
    - MEDUSA_PUBLISHABLE_KEY must be set (for API authentication).
    - MEDUSA_BASE_URL must point to a Medusa backend with the Store API enabled.
    """
    if not PUBLISHABLE_KEY:
        raise RuntimeError("MEDUSA_PUBLISHABLE_KEY env var not set")

    url = f"{BASE_URL}/store/products?limit={limit}&offset={offset}"

    headers = {
        "x-publishable-api-key": PUBLISHABLE_KEY
    }

    try:
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error connecting to Medusa Store API: {e}")
