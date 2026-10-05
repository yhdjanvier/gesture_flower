"""ICE (STUN/TURN) servers for WebRTC. Cloud hosting usually needs a TURN relay for the browser webcam.

Configure ONE of these in Streamlit Cloud -> app Settings -> Secrets (or .streamlit/secrets.toml locally):
  METERED_DOMAIN = "yourname.metered.live"      METERED_API_KEY = "..."      (free Open Relay / Metered TURN)
  TWILIO_ACCOUNT_SID = "AC..."                  TWILIO_AUTH_TOKEN = "..."    (Twilio Network Traversal; needs `twilio` in requirements)
Without secrets it falls back to Google STUN only (works on many home networks, often fails on cloud/school networks).
"""
from __future__ import annotations

from typing import Mapping

STUN_ONLY = [{"urls": ["stun:stun.l.google.com:19302"]}]


def get_ice_servers(secrets: Mapping) -> list:
    domain, key = secrets.get("METERED_DOMAIN"), secrets.get("METERED_API_KEY")
    if domain and key:
        try:
            import requests
            r = requests.get(f"https://{domain}/api/v1/turn/credentials", params={"apiKey": key}, timeout=8)
            r.raise_for_status()
            servers = r.json()
            if isinstance(servers, list) and servers:
                return servers
        except Exception as exc:
            print(f"[ice] Metered TURN failed: {exc}")
    sid, token = secrets.get("TWILIO_ACCOUNT_SID"), secrets.get("TWILIO_AUTH_TOKEN")
    if sid and token:
        try:
            from twilio.rest import Client
            return Client(sid, token).tokens.create().ice_servers
        except Exception as exc:
            print(f"[ice] Twilio TURN failed: {exc}")
    return STUN_ONLY + [{"urls": ["stun:stun1.l.google.com:19302"]}]


def has_turn(servers: list) -> bool:
    """True if at least one TURN (relay) server is configured."""
    for s in servers:
        urls = s.get("urls") or s.get("url") or []
        urls = [urls] if isinstance(urls, str) else urls
        if any(str(u).startswith(("turn:", "turns:")) for u in urls):
            return True
    return False
