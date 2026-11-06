#!/usr/bin/env python3
"""
Description: Drop non-escalation labels whose reason_evidence quotes are not note substrings.
"""

# Imports
import re

# Constants
CCT = "continue_current_therapy"
REASON_EVIDENCE_KEY = "reason_evidence"
WHITESPACE_RE = re.compile(r"\s+")


def normalize_reasons(reasons):
    if reasons is None:
        return []
    if isinstance(reasons, str):
        return [reasons] if reasons else []
    out = []
    seen = set()
    for reason in reasons:
        if not reason or reason in seen:
            continue
        seen.add(reason)
        out.append(reason)
    return out


def normalize_evidence_map(raw):
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, value in raw.items():
        label = str(key).strip()
        if not label:
            continue
        quotes = []
        if isinstance(value, str):
            text = value.strip()
            if text:
                quotes.append(text)
        elif isinstance(value, list):
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    quotes.append(text)
        elif value is not None:
            text = str(value).strip()
            if text:
                quotes.append(text)
        if quotes:
            out[label] = quotes
    return out


def collapse_ws(text):
    return WHITESPACE_RE.sub(" ", text).strip()


def quote_in_note(quote, note):
    if not quote or not note:
        return False
    if quote in note:
        return True

    # Strip wrapping quotes
    trimmed = quote.strip().strip('"').strip("'").strip()
    if trimmed and trimmed in note:
        return True

    # Whitespace tolerant
    note_norm = collapse_ws(note)
    for candidate in (quote, trimmed):
        if not candidate:
            continue
        if collapse_ws(candidate) in note_norm:
            return True
    return False


def label_grounded(label, evidence_map, note):
    if label == CCT:
        return True
    quotes = evidence_map.get(label) or []
    for quote in quotes:
        if quote_in_note(quote, note):
            return True
    return False


def apply_evidence_span_gate(prediction, text=None):
    if not isinstance(prediction, dict):
        return prediction

    reasons = normalize_reasons(prediction.get("llt_reasons"))
    note = text if text is not None else prediction.get("text") or ""
    evidence_map = normalize_evidence_map(prediction.get(REASON_EVIDENCE_KEY))

    kept = []
    for reason in reasons:
        if label_grounded(reason, evidence_map, note):
            kept.append(reason)

    # CCT fallback when empty
    if not kept and reasons:
        kept.append(CCT)

    out = dict(prediction)
    out["llt_reasons"] = kept
    if "other" not in kept:
        out["other_reasons"] = {}
    else:
        out["other_reasons"] = prediction.get("other_reasons") or {}
    out.pop(REASON_EVIDENCE_KEY, None)
    return out
