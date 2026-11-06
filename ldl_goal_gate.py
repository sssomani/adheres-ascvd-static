#!/usr/bin/env python3
"""
Description: Drop LDL goal labels inconsistent with model-assessed ldl_goal_status.
"""

# Imports
from evidence_span_gate import normalize_reasons

# Constants
CCT = "continue_current_therapy"
LDL_AT_GOAL = "ldl_at_goal"
LDL_CLOSE = "ldl_close_to_goal"
STATUS_FOR_LABEL = {
    LDL_AT_GOAL: {"at_goal"},
    LDL_CLOSE: {"close"},
}


def apply_ldl_goal_gate(prediction):
    if not isinstance(prediction, dict):
        return prediction

    reasons = normalize_reasons(prediction.get("llt_reasons"))
    status = str(prediction.get("ldl_goal_status") or "unknown").strip().lower()
    kept = []

    for reason in reasons:
        allowed = STATUS_FOR_LABEL.get(reason)
        if allowed is None:
            kept.append(reason)
            continue
        if status in allowed:
            kept.append(reason)

    if not kept and reasons:
        kept.append(CCT)

    out = dict(prediction)
    out["llt_reasons"] = kept
    if "other" not in kept:
        out["other_reasons"] = {}
    return out
