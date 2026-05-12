"""
pinterest_auth.py
-----------------
OAuth helper script for Pinterest API.

Use this once to obtain a fresh PINTEREST_ACCESS_TOKEN.
The script reads CLIENT_ID, CLIENT_SECRET, and AUTH_CODE from the
pipeline/.env file, exchanges the code for a token, and prints it
so you can update the same .env file.
"""

import requests
import base64
import os
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 1. Load environment variables from pipeline/.env
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(script_dir, '..')
env_path = os.path.join(pipeline_dir, '.env')

if not os.path.exists(env_path):
    raise FileNotFoundError(
        f"Missing .env file at {env_path}. "
        "Create it from .env.example and fill in your Pinterest credentials."
    )

load_dotenv(dotenv_path=env_path)

CLIENT_ID = os.getenv("PINTEREST_CLIENT_ID")
CLIENT_SECRET = os.getenv("PINTEREST_CLIENT_SECRET")
AUTH_CODE = os.getenv("PINTEREST_AUTH_CODE")
REDIRECT_URI = "http://localhost/"   # standard for local OAuth flows

if not all([CLIENT_ID, CLIENT_SECRET, AUTH_CODE]):
    raise ValueError(
        "Missing one or more required variables: "
        "PINTEREST_CLIENT_ID, PINTEREST_CLIENT_SECRET, PINTEREST_AUTH_CODE"
    )

# ----------------------------------------------------------------------
# 2. Exchange authorization code for a new access token
# ----------------------------------------------------------------------
def exchange_code_for_token():
    url = "https://api.pinterest.com/v5/oauth/token"

    # Base64 encode client id and secret for the Authorization header
    auth_str = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = {
        "grant_type": "authorization_code",
        "code": AUTH_CODE,
        "redirect_uri": REDIRECT_URI
    }

    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        token_data = response.json()
        print("\n✅ SUCCESS! Your New Access Token:")
        print(token_data.get("access_token"))
        print("\nCopy the token above and paste it into your pipeline/.env file:")
        print("  PINTEREST_ACCESS_TOKEN=<token>")
    else:
        print(f"\n❌ Error {response.status_code}: {response.text}")

# ----------------------------------------------------------------------
# 3. Execute the exchange when the script is run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    exchange_code_for_token()