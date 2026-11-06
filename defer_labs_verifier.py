#!/usr/bin/env python3
"""
Description: Drop-only LLM verifier for defer_after_labs false positives.
"""

# Imports
import dspy
from typing import Literal
from evidence_span_gate import CCT, normalize_evidence_map, normalize_reasons

# Constants
DEFER_AFTER_LABS = "defer_after_labs"
REASON_EVIDENCE_KEY = "reason_evidence"
KeepLabs = Literal["yes", "no"]


class DeferAfterLabsVerify(dspy.Signature):
    """Verify whether defer_after_labs is justified for this note.

    Keep ("yes") ONLY if the assessment/plan explicitly conditions a FUTURE LLT change
    (titration, adding/switching an agent, or reassessing the medication) on the result
    of a specific pending lab/imaging test — e.g. "will titrate statin based on results",
    "reassess need for atorvastatin 80 after labs", "if LDL still high on repeat panel,
    add ezetimibe", "will consider changing her statin pending LFTs".

    Answer "no" when the plan merely CONTINUES the current LLT and orders/repeats a lipid
    panel for routine surveillance without tying it to an LLT change (e.g. "continue
    atorvastatin 80 mg, recheck lipids in 3-6 months", "cont statin - repeat lipid panel",
    "check labs with lipids in 6 months"). Also answer "no" for bare lipid-panel orders,
    screening labs, "labs today", or future lab orders without LLT-decision language.
    Continuing therapy while monitoring lipids is continue_current_therapy, not defer.
    Prefer precision: when unsure, answer "no".
    """

    text: str = dspy.InputField(desc="full clinical note")
    candidate_quotes: str = dspy.InputField(
        desc="verbatim quote spans the model cited for defer_after_labs, or 'none'"
    )
    keep_defer_after_labs: KeepLabs = dspy.OutputField(
        desc="yes to keep defer_after_labs; no to drop it"
    )


def apply_defer_labs_verifier(prediction, text=None, verifier=None):
    if verifier is None or not isinstance(prediction, dict):
        return prediction

    reasons = normalize_reasons(prediction.get("llt_reasons"))
    if DEFER_AFTER_LABS not in reasons:
        return prediction

    evidence_map = normalize_evidence_map(prediction.get(REASON_EVIDENCE_KEY))
    quotes = list(evidence_map.get(DEFER_AFTER_LABS) or [])
    quote_blob = " | ".join(q.strip() for q in quotes if q and str(q).strip()) or "none"
    note = text if text is not None else prediction.get("text") or ""

    try:
        result = verifier(text=note, candidate_quotes=quote_blob)
        keep = str(getattr(result, "keep_defer_after_labs", "") or "").strip().lower()
    except Exception:
        return prediction

    if keep in {"yes", "y", "true", "1"}:
        return prediction

    # Drop labs only
    kept = [r for r in reasons if r != DEFER_AFTER_LABS]
    if not kept:
        kept = [CCT]
    out = dict(prediction)
    out["llt_reasons"] = kept
    if REASON_EVIDENCE_KEY in out and isinstance(out[REASON_EVIDENCE_KEY], dict):
        evidence = dict(out[REASON_EVIDENCE_KEY])
        evidence.pop(DEFER_AFTER_LABS, None)
        out[REASON_EVIDENCE_KEY] = evidence
    return out
