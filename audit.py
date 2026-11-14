"""Hash-chained audit log.

Every tool call, tool result, model step and operator action is appended to a
chain where each entry carries the digest of its predecessor. Tamper with any
entry and verification fails. This is the smallest useful answer to "prove what
the agent did" in a review.

This is not a ledger, not signed, and not a substitute for a SIEM. It is a
tamper-evident local record you can hand to an evaluator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone


def _digest(entry: dict) -> str:
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _short(value) -> str:
    blob = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class AuditLog:
    """Append-only, hash-chained log of agent activity."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._prev = "GENESIS"

    def record(self, kind: str, actor: str, payload: dict) -> dict:
        entry = {
            "seq": len(self.entries) + 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "kind": kind,
            "actor": actor,
            "payload": payload,
            "prev": self._prev,
        }
        entry["digest"] = _digest(entry)
        self._prev = entry["digest"]
        self.entries.append(entry)
        return entry

    def record_tool(self, tool: str, arguments: dict, result=None, error: str | None = None) -> dict:
        payload = {"tool": tool, "arguments": arguments}
        if error is not None:
            payload["error"] = error
        if result is not None:
            payload["result_digest"] = _short(result)
            payload["result_bytes"] = len(json.dumps(result, default=str))
        return self.record("tool_error" if error else "tool_result", "agent", payload)

    def verify(self) -> tuple[bool, str]:
        """Walk the chain. Returns (ok, message)."""
        prev = "GENESIS"
        for i, entry in enumerate(self.entries):
            body = {k: v for k, v in entry.items() if k != "digest"}
            if body["prev"] != prev:
                return False, f"entry {i + 1}: broken link to predecessor"
            if _digest(body) != entry["digest"]:
                return False, f"entry {i + 1}: payload modified after append"
            prev = entry["digest"]
        return True, f"{len(self.entries)} entries verified"

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e["kind"]] = counts.get(e["kind"], 0) + 1
        ok, message = self.verify()
        return {"entries": len(self.entries), "kinds": counts,
                "chain_ok": ok, "message": message, "head": self._prev}

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e, sort_keys=True) for e in self.entries)
