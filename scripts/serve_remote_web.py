"""Start the authenticated simulator HTTP API without putting secrets in argv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--check-health", action="store_true")
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32 or any(character.isspace() for character in token):
        parser.error("The private API token must contain at least 32 non-whitespace characters.")
    if args.check_health:
        # The boot service needs a bounded, proxy-free check in session zero.
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/api/health",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "HeteroLLMLab/4.0.0"},
        )
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2) as response:
                healthy = json.load(response).get("ok") is True
        except urllib.error.HTTPError as error:
            try:
                error_code = json.load(error).get("error", {}).get("code")
            except (ValueError, AttributeError):
                error_code = "non_json_health_response"
            print(json.dumps({"ok": False, "http_status": error.code, "code": error_code}))
            raise SystemExit(1) from None
        except (OSError, ValueError):
            print(json.dumps({"ok": False, "error": "local_health_unavailable"}))
            raise SystemExit(1) from None
        print(json.dumps({"ok": healthy}))
        raise SystemExit(0 if healthy else 1)
    from heterollm_sim.web import serve

    serve("127.0.0.1", args.port, open_browser=False, remote_access_token=token,
          trusted_origins=("https://zh.gripe", "https://www.zh.gripe"))


if __name__ == "__main__":
    main()
