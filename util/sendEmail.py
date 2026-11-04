#!/usr/bin/env python3
"""Send Opera completion email via Resend API (stdlib only).

Set RESEND_API_KEY and OPERA_EMAIL_TO in the environment to receive an email when a job ends;
without them nothing is sent.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.resend.com/emails"
FROM_ADDR = "onboarding@resend.dev"
SUBJECT = "Opera: code execution complete"


def send_email(api_key: str, to_addr: str, exp_log: str | None = None) -> dict:
    html = "<p>Code execution complete!</p>"
    payload: dict = {
        "from": FROM_ADDR,
        "to": [to_addr],
        "subject": SUBJECT,
        "html": html,
    }

    if exp_log:
        payload["html"] = html + f"<p>Log file: <code>{exp_log}</code></p>"
        log_path = Path(exp_log)
        if log_path.is_file():
            content_b64 = base64.b64encode(log_path.read_bytes()).decode("ascii")
            payload["attachments"] = [
                {"filename": log_path.name, "content": content_b64}
            ]

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Opera-sendEmail/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send Opera completion email via Resend")
    parser.add_argument(
        "exp_log",
        nargs="?",
        default="",
        help="Optional path to log file to mention/attach",
    )
    args = parser.parse_args(argv)
    api_key = os.environ.get("RESEND_API_KEY")
    to_addr = os.environ.get("OPERA_EMAIL_TO")
    if not api_key or not to_addr:
        print("No email sent (RESEND_API_KEY or OPERA_EMAIL_TO is not set).")
        return 0
    try:
        result = send_email(api_key, to_addr, args.exp_log or None)
        print(f"Resend email accepted: {result}")
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"Resend HTTP error {exc.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - keep experiment runners alive via caller
        print(f"Resend email failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
