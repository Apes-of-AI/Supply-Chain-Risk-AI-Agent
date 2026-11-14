"""Diagnose model reachability for your HF token.
    python check_providers.py
    python check_providers.py openai/gpt-oss-20b Qwen/Qwen3-8B
Prints the token identity, the cheapest tool-capable models the router is
currently serving, and a live 1-token probe against each candidate so you can
tell "not served" apart from "not authorised" apart from "no billing".
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

ROUTER_MODELS_URL = "https://router.huggingface.co/v1/models"

CANDIDATES = [
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "meta-llama/Llama-3.3-70B-Instruct",
]


def catalogue() -> list[dict]:
    request = urllib.request.Request(ROUTER_MODELS_URL, headers={"User-Agent": "scr-check/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8")).get("data", [])


def main(argv: list[str]) -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"HF_TOKEN present: {bool(token)}")

    if token:
        try:
            from huggingface_hub import HfApi
            who = HfApi().whoami(token=token)
            name = who.get("name") or who.get("user", {}).get("name", "?")
            print(f"token identity : {name} (plan: {who.get('plan', 'unknown')})")
        except Exception as exc:  # noqa: BLE001
            print(f"token rejected : {type(exc).__name__}: {str(exc)[:160]}")

    try:
        models = catalogue()
    except Exception as exc:  # noqa: BLE001
        print(f"\ncould not read the router catalogue: {type(exc).__name__}: {exc}")
        return 1

    rows = []
    for entry in models:
        live = [
            p for p in entry.get("providers", [])
            if p.get("status") == "live" and p.get("supports_tools")
        ]
        if not live:
            continue
        price = min(
            (p.get("pricing", {}).get("input") for p in live if p.get("pricing", {}).get("input") is not None),
            default=None,
        )
        rows.append((price if price is not None else 9e9, entry["id"], len(live)))
    rows.sort()

    print(f"\n{len(rows)} models currently served live with tool calling.")
    print(f"{'in $/M':>8}  {'providers':>9}  model (cheapest 15)")
    for price, mid, count in rows[:15]:
        shown = f"{price:.2f}" if price < 9e9 else "n/a"
        print(f"{shown:>8}  {count:>9}  {mid}")

    targets = argv[1:] or CANDIDATES
    print("\nlive probe (1 token each):")
    failures = 0
    from huggingface_hub import InferenceClient
    for mid in targets:
        try:
            InferenceClient(model=mid).chat_completion(
                messages=[{"role": "user", "content": "ping"}], max_tokens=1
            )
            print(f"  OK    {mid}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {mid}  ->  {type(exc).__name__}: {str(exc)[:140]}")

    print("\nNext step: put a working id in SMOLAGENTS_MODEL_ID, or type it into the "
          "Model override dropdown in the Space.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
