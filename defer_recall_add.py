#!/usr/bin/env python3
"""
Description: Add-only LLM verifier for defer_after_labs false negatives. Runs when the
main pass did NOT select defer_after_labs; adds it only when assessment/plan clearly
conditions a future LLT change on pending labs.
"""

# Imports
import dspy
from typing import Literal
from evidence_span_gate import CCT, normalize_evidence_map, normalize_reasons

# Constants
DEFER_AFTER_LABS = "defer_after_labs"
REASON_EVIDENCE_KEY = "reason_evidence"
AddLabs = Literal["yes", "no"]


class DeferAfterLabsAdd(dspy.Signature):
    """Check whether defer_after_labs was missed for this note.

    Answer "yes" ONLY if assessment/plan explicitly conditions a FUTURE LLT change
    (titration, add-on, switch, reassess medication) on pending lab/imaging results,
    but the current label set omitted defer_after_labs. Examples: "will titrate statin
    based on results", "reassess need for atorvastatin 80 after labs", "if LDL still
    high on repeat panel, add ezetimibe".

    Answer "no" for routine continue statin + recheck lipids surveillance, bare lipid
    orders, or when defer_after_labs is already present or not applicable.
    """

    text: str = dspy.InputField(desc="full clinical note")
    current_reasons: str = dspy.InputField(desc="comma-separated current llt_reasons")
    add_defer_after_labs: AddLabs = dspy.OutputField(
        desc="yes to add defer_after_labs; no to leave unchanged"
    )


def apply_defer_recall_add(prediction, text=None, adder=None):
    if adder is None or not isinstance(prediction, dict):
        return prediction

    reasons = normalize_reasons(prediction.get("llt_reasons"))
    if DEFER_AFTER_LABS in reasons:
        return prediction

    note = text if text is not None else prediction.get("text") or ""
    reason_blob = ", ".join(reasons) or "none"

    try:
        result = adder(text=note, current_reasons=reason_blob)
        add = str(getattr(result, "add_defer_after_labs", "") or "").strip().lower()
    except Exception:
        return prediction

    if add not in {"yes", "y", "true", "1"}:
        return prediction

    out = dict(prediction)
    out["llt_reasons"] = reasons + [DEFER_AFTER_LABS]
    evidence = normalize_evidence_map(out.get(REASON_EVIDENCE_KEY))
    if DEFER_AFTER_LABS not in evidence:
        evidence[DEFER_AFTER_LABS] = ["pending labs linked to LLT decision"]
    out[REASON_EVIDENCE_KEY] = evidence
    return out
