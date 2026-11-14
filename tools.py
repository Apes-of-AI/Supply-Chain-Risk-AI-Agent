"""Tool layer.
Two responsibilities:
  1. A registry of read-only functions over the dataset, each invocation
     audited. Every function is side-effect free. `draft_course_of_action`
     returns a *proposal*; nothing in this file mutates state.
  2. Thin `@tool` wrappers so smolagents can call the same functions without
     duplicating logic.
The allowlist is the registry itself: an agent cannot call what is not here.
"""

from __future__ import annotations

import hashlib
import json

import data_layer as dl
from audit import AuditLog

# Action vocabulary. Each entry documents who owns it, how long it takes and
# what approval it needs. The agent may propose these strings, never invent new
# ones, and never execute any of them.
ACTIONS: dict[str, dict] = {
    "expedite_from_alternate": {
        "label": "Expedite from alternate source",
        "owner": "Item manager",
        "lead_time_days": 6,
        "cost": "premium freight, roughly +8-14% unit cost",
        "approval": "Item manager",
        "addresses": ["exposure", "coverage_gap"],
    },
    "increase_safety_stock": {
        "label": "Increase safety stock",
        "owner": "Demand planning",
        "lead_time_days": 45,
        "cost": "carrying cost, roughly 18%/yr of inventory value",
        "approval": "Supply chain lead",
        "addresses": ["coverage_gap"],
    },
    "pre_position_stock": {
        "label": "Pre-position stock away from hazard",
        "owner": "Distribution operations",
        "lead_time_days": 14,
        "cost": "one-time relocation plus dual storage",
        "approval": "Distribution operations lead",
        "addresses": ["exposure"],
    },
    "qualify_second_source": {
        "label": "Qualify a second source",
        "owner": "Engineering and contracting",
        "lead_time_days": 180,
        "cost": "qualification test campaign",
        "approval": "Program office",
        "addresses": ["sole_source"],
    },
    "defer_non_critical_demand": {
        "label": "Defer non-critical demand",
        "owner": "Requirements",
        "lead_time_days": 0,
        "cost": "readiness risk at using units",
        "approval": "Requirements owner",
        "addresses": ["coverage_gap"],
    },
}

REGISTRY: dict[str, dict] = {}


def _register(name: str, description: str, read_only: bool = True):
    def deco(fn):
        REGISTRY[name] = {
            "name": name,
            "description": description,
            "read_only": read_only,
            "fn": fn,
        }
        return fn
    return deco


def invoke(audit: AuditLog, name: str, **kwargs):
    """Single choke point for every tool call. Audits before and after."""
    if name not in REGISTRY:
        audit.record("policy_violation", "runtime", {"reason": "tool not in allowlist", "tool": name})
        raise PermissionError(f"Tool '{name}' is not in the allowlist.")
    spec = REGISTRY[name]
    audit.record("tool_call", "agent", {"tool": name, "arguments": kwargs, "read_only": spec["read_only"]})
    try:
        result = spec["fn"](**kwargs)
    except dl.DataUnavailable as exc:
        audit.record_tool(name, kwargs, error=str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as an observation
        audit.record_tool(name, kwargs, error=f"{type(exc).__name__}: {exc}")
        raise
    audit.record_tool(name, kwargs, result=result)
    return result


# --------------------------------------------------------------------------
# plain functions (also used by the deterministic baseline and the evals)
# --------------------------------------------------------------------------

@_register("list_parts", "List candidate parts, highest criticality first.")
def list_parts(min_criticality: int = 1) -> list:
    rows = [
        {"part_id": p["part_id"], "name": p["name"], "criticality": p["criticality"],
         "unit_cost": p["unit_cost"], "annual_demand": p["annual_demand"]}
        for p in dl.parts() if p["criticality"] >= min_criticality
    ]
    return sorted(rows, key=lambda r: (-r["criticality"], r["part_id"]))


@_register("part_profile", "Master record for one part, including authorized sources.")
def part_profile(part_id: str) -> dict:
    part = dl.get_part(part_id)
    suppliers = [dl.get_supplier(s) for s in part["authorized_suppliers"]]
    return {
        "part": part,
        "suppliers": [
            {k: s[k] for k in ("supplier_id", "name", "city", "region", "tier",
                               "lead_time_days", "on_time_delivery_pct", "financial_health")}
            | {"hazard_hits": len(dl.exposure(s["supplier_id"]))}
            for s in suppliers
        ],
    }


@_register("supplier_profile", "Supplier record plus any active hazard exposure.")
def supplier_profile(supplier_id: str) -> dict:
    supplier = dl.get_supplier(supplier_id)
    return {"supplier": supplier, "active_hazard_hits": dl.exposure(supplier_id)}


@_register("inventory_position", "On-hand and in-transit quantity by site, in units and days.")
def inventory_position(part_id: str) -> dict:
    return dl.days_of_supply(part_id)


@_register("active_hazards", "Active hazards within radius_km of a coordinate.")
def active_hazards(lat: float, lon: float, radius_km: float = 500.0) -> list:
    out = []
    for h in dl.hazards():
        d = dl.haversine_km(lat, lon, h["lat"], h["lon"])
        if d <= float(radius_km):
            out.append({"hazard_id": h["hazard_id"], "hazard_type": h["hazard_type"],
                        "severity": h["severity"], "distance_km": round(d, 1),
                        "radius_km": h["radius_km"], "region": h["region"],
                        "starts": h["starts"], "ends": h["ends"],
                        "source": h["source"], "note": h["note"]})
    return sorted(out, key=lambda x: (-x["severity"], x["distance_km"]))


@_register("assess_part_risk", "Composite risk score with weighted components and cited drivers.")
def assess_part_risk(part_id: str) -> dict:
    return dl.part_risk(part_id)


@_register("draft_course_of_action", "Propose mitigations for one part. Does not execute anything.")
def draft_course_of_action(part_id: str, actions: list) -> dict:
    risk = dl.part_risk(part_id)
    if isinstance(actions, str):
        actions = [actions]
    unknown = [a for a in actions if a not in ACTIONS]
    if unknown:
        raise ValueError(
            f"Unknown action(s): {unknown}. Permitted values: {sorted(ACTIONS)}."
        )
    basis_by_component: dict[str, str] = {}
    for d in risk["drivers"]:
        basis_by_component.setdefault(d["component"], d["statement"])

    proposal = []
    for name in actions:
        spec = ACTIONS[name]
        proposal.append({
            "action": name,
            "label": spec["label"],
            "owner": spec["owner"],
            "lead_time_days": spec["lead_time_days"],
            "cost": spec["cost"],
            "approval": spec["approval"],
            "addresses": spec["addresses"],
            "basis": " ".join(basis_by_component[c] for c in spec["addresses"]
                              if c in basis_by_component) or "No driver cited for this action.",
        })

    fingerprint = hashlib.sha256(
        json.dumps([part_id, sorted(actions)], sort_keys=True).encode()
    ).hexdigest()[:8]
    return {
        "status": "PROPOSED_AWAITING_APPROVAL",
        "proposal_id": f"COA-{part_id}-{fingerprint}",
        "part_id": part_id,
        "part_name": risk["name"],
        "risk_score": risk["score"],
        "risk_band": risk["band"],
        "actions": proposal,
        "note": ("No system of record was modified. Execution requires a named human "
                 "approver acting outside this agent."),
    }


# --------------------------------------------------------------------------
# smolagents wrappers
# --------------------------------------------------------------------------

def build_smol_tools(audit: AuditLog) -> list:
    """Return @tool-wrapped callables bound to one audit log."""
    from smolagents import tool

    @tool
    def list_parts_tool(min_criticality: int = 1) -> list:
        """List candidate parts, highest criticality first.
        Args:
            min_criticality: Minimum item criticality to include, 1 to 5.
        """
        return invoke(audit, "list_parts", min_criticality=min_criticality)

    @tool
    def part_profile_tool(part_id: str) -> dict:
        """Master record for one part, including its authorized sources.
        Args:
            part_id: Part identifier such as PN-1042.
        """
        return invoke(audit, "part_profile", part_id=part_id)

    @tool
    def supplier_profile_tool(supplier_id: str) -> dict:
        """Supplier record plus any active hazard exposure.
        Args:
            supplier_id: Supplier identifier such as SUP-004.
        """
        return invoke(audit, "supplier_profile", supplier_id=supplier_id)

    @tool
    def inventory_position_tool(part_id: str) -> dict:
        """On-hand and in-transit quantity by site, in units and days.
        Args:
            part_id: Part identifier such as PN-1042.
        """
        return invoke(audit, "inventory_position", part_id=part_id)

    @tool
    def active_hazards_tool(lat: float, lon: float, radius_km: float = 500.0) -> list:
        """Active hazards within a radius of a coordinate.
        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            radius_km: Search radius in kilometres.
        """
        return invoke(audit, "active_hazards", lat=lat, lon=lon, radius_km=radius_km)

    @tool
    def assess_part_risk_tool(part_id: str) -> dict:
        """Composite risk score for one part, with weighted components and cited drivers.
        Args:
            part_id: Part identifier such as PN-1042.
        """
        return invoke(audit, "assess_part_risk", part_id=part_id)

    @tool
    def draft_course_of_action_tool(part_id: str, actions: list) -> dict:
        """Propose mitigations for one part. Returns a proposal only; executes nothing.
        Args:
            part_id: Part identifier such as PN-1042.
            actions: List of action names drawn from the permitted vocabulary, e.g.
                ["expedite_from_alternate", "increase_safety_stock"].
        """
        return invoke(audit, "draft_course_of_action", part_id=part_id, actions=actions)

    return [list_parts_tool, part_profile_tool, supplier_profile_tool,
            inventory_position_tool, active_hazards_tool, assess_part_risk_tool,
            draft_course_of_action_tool]


def action_vocabulary() -> str:
    return ", ".join(sorted(ACTIONS))
