#!/usr/bin/env python3
"""
Description: Boundary reason reclassification. When the main non-escalation
pass selects any boundary label (defer_after_labs / defer_future_visit /
defer_alternate_clinician / other), a focused few-shot disambiguator re-derives the
correct subset of boundary labels from the note, using TRAIN-mined contrast demos.
Non-boundary labels from the main pass are preserved; boundary labels are replaced.
Runs before apply_nonescalation_gates so the deterministic gates still validate output.
"""

# Imports
import json
from pathlib import Path
from typing import List, Literal
import dspy
from evidence_span_gate import CCT, normalize_reasons, normalize_evidence_map

# Constants
BOUNDARY_LABELS = ["continue_current_therapy", "defer_after_labs", "defer_future_visit", "defer_alternate_clinician", "other"]
TRIGGER_LABELS = {"defer_after_labs", "defer_future_visit", "defer_alternate_clinician", "other"}
BoundaryLabel = Literal[
    "continue_current_therapy",
    "defer_after_labs",
    "defer_future_visit",
    "defer_alternate_clinician",
    "other",
]



class BoundaryReassert(dspy.Signature):
    """Re-classify ONLY the boundary reasons for non-escalation of lipid-lowering therapy (LLT): continue_current_therapy, defer_after_labs, defer_future_visit, defer_alternate_clinician, other. Do not output any other category. Base the decision on the current visit's assessment/impression and plan.

    Strict definitions:
    - defer_after_labs: select ONLY when the plan explicitly conditions a FUTURE LLT change (titration, add-on, switch, or reassessing the medication) on a specific pending lab/imaging result — e.g. "lipid panel ordered, will titrate statin accordingly", "reassess need for atorvastatin 80 after labs", "if LDL still high on repeat panel, add ezetimibe". Routine "continue current statin + recheck/repeat lipids in N months" surveillance is continue_current_therapy, NOT defer_after_labs.
    - defer_future_visit: deferring the LLT decision to a future clinic visit with explicit documented rationale, including conditional future LLT plans timed to a later visit ("if LDL remains >70 in 6 months, would favor adding ezetimibe"). Routine return-to-clinic alone is not enough.
    - defer_alternate_clinician: explicitly deferring the LLT decision to another clinician.
    - other: a genuinely novel, explicitly documented rationale that fits none of the categories; be extremely conservative — almost always this should be continue_current_therapy instead.
    - continue_current_therapy: continuation of current therapy without a clearly acceptable, specific deferral reason. This is the correct default when the plan simply continues the statin and monitors lipids.

    Return corrected_boundary_labels as the subset of these five labels the note supports (use continue_current_therapy if none of the specific deferral reasons apply). For each selected label except continue_current_therapy, provide boundary_evidence: a verbatim quote span copied exactly from the note.
    """

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField(desc="medical document")
    corrected_boundary_labels: List[BoundaryLabel] = dspy.OutputField(
        format=list, desc="corrected subset of the five boundary labels"
    )
    boundary_evidence: dict[str, list[str]] = dspy.OutputField(
        desc="verbatim note quote spans per selected boundary label (except continue_current_therapy)"
    )


def load_boundary_demos(path=None):
    """Load explicitly supplied examples; no clinical demonstrations are bundled."""
    if path is None:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("examples must be a JSON array")
    demos = []
    for d in raw:
        if not isinstance(d, dict) or not isinstance(d.get("text"), str) or not d["text"].strip():
            raise ValueError("each example requires nonempty text")
        labels = d.get("corrected_boundary_labels")
        evidence = d.get("boundary_evidence") or {}
        if not isinstance(labels, list) or not all(isinstance(x, str) and x in BOUNDARY_LABELS for x in labels):
            raise ValueError("example labels must be a list of boundary labels")
        if not isinstance(evidence, dict) or not all(
            k in BOUNDARY_LABELS and isinstance(v, list) and all(isinstance(q, str) for q in v)
            for k, v in evidence.items()
        ):
            raise ValueError("example evidence must map boundary labels to quote lists")
        ex = dspy.Example(
            context="",
            text=d["text"],
            corrected_boundary_labels=d["corrected_boundary_labels"],
            boundary_evidence=d.get("boundary_evidence") or {},
        ).with_inputs("context", "text")
        demos.append(ex)
    return demos


def should_reassert(reasons):
    return any(r in TRIGGER_LABELS for r in reasons)


def apply_boundary_reassert(prediction, text=None, context="", reasoner=None):
    if reasoner is None or not isinstance(prediction, dict):
        return prediction

    reasons = normalize_reasons(prediction.get("llt_reasons"))
    if not should_reassert(reasons):
        return prediction

    note = text if text is not None else prediction.get("text") or ""
    try:
        result = reasoner(context=context, text=note)
        corrected = normalize_reasons(getattr(result, "corrected_boundary_labels", None))
        new_evidence = normalize_evidence_map(getattr(result, "boundary_evidence", None))
    except Exception:
        return prediction

    corrected = [c for c in corrected if c in BOUNDARY_LABELS]
    nonboundary = [r for r in reasons if r not in BOUNDARY_LABELS]
    merged = nonboundary + [c for c in corrected if c not in nonboundary]
    if not merged:
        merged = [CCT]

    # de-dup preserving order
    seen = set()
    final = []
    for r in merged:
        if r not in seen:
            seen.add(r)
            final.append(r)

    out = dict(prediction)
    out["llt_reasons"] = final

    # Merge boundary evidence into reason_evidence so the span gate can validate.
    evidence = normalize_evidence_map(prediction.get("reason_evidence"))
    for label in BOUNDARY_LABELS:
        evidence.pop(label, None)
    for label, quotes in new_evidence.items():
        if label in BOUNDARY_LABELS and label != CCT:
            evidence[label] = quotes
    out["reason_evidence"] = evidence
    return out
