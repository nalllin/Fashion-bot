import os
import requests

MEDUSA_BASE_URL = os.environ.get("MEDUSA_BASE_URL", "http://localhost:9000")
MEDUSA_TOKEN = os.environ.get("MEDUSA_TOKEN")  # Your long JWT

def fetch_recommendations(product_id: str):
    """
    Fetch recommended products from Medusa Vendor API.
    """
    if not MEDUSA_TOKEN:
        raise RuntimeError("MEDUSA_TOKEN env var is not set")

    url = f"{MEDUSA_BASE_URL}/vendor/products/{product_id}/recommendation"

    headers = {
        "Authorization": f"Bearer {MEDUSA_TOKEN}",
        "Content-Type": "application/json"
    }

    print("Calling:", url)
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()
