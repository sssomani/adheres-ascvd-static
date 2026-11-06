#!/usr/bin/env python3
"""
Description: Filter "other" non-escalation labels to supported themes
(no-ASCVD-yet / derisking / CAC / polypharmacy). Keep "other" only when the
model's free-text rationale matches those themes; otherwise drop it (falling back to
continue_current_therapy when it was the sole reason).
"""

# Imports
import re
from evidence_span_gate import CCT, normalize_reasons

# Constants
OTHER = "other"
GENUINE_OTHER_RE = re.compile(
    r"(no\s+(known\s+)?ascvd|no\s+cad|not\s+(yet\s+)?(for\s+)?secondary\s+prevention|"
    r"derisk|de-risk|\bcac\b|calcium\s+score|coronary\s+calcium|polypharmacy)",
    re.I,
)


def other_text_blob(other_reasons):
    if not isinstance(other_reasons, dict):
        if isinstance(other_reasons, str):
            return other_reasons
        return ""
    parts = []
    for key, value in other_reasons.items():
        parts.append(str(key))
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        else:
            parts.append(str(value))
    return " ".join(parts)


def apply_other_suppression_gate(prediction):
    if not isinstance(prediction, dict):
        return prediction

    reasons = normalize_reasons(prediction.get("llt_reasons"))
    if OTHER not in reasons:
        return prediction

    blob = other_text_blob(prediction.get("other_reasons"))
    keep_other = bool(GENUINE_OTHER_RE.search(blob))

    if keep_other:
        return prediction

    kept = [r for r in reasons if r != OTHER]
    if not kept:
        kept = [CCT]
    out = dict(prediction)
    out["llt_reasons"] = kept
    out["other_reasons"] = {}
    return out
