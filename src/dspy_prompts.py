#!/usr/bin/env python3
"""
Description: Prompts and routing for LLT decisions, reasons, and LDL targets.
"""

# Imports
import dspy
from typing import List, Literal
from defer_labs_verifier import DeferAfterLabsVerify, apply_defer_labs_verifier
from defer_recall_add import DeferAfterLabsAdd, apply_defer_recall_add
from evidence_span_gate import apply_evidence_span_gate
from ldl_goal_gate import apply_ldl_goal_gate
from other_suppression_gate import apply_other_suppression_gate
from boundary_reassert import BoundaryReassert, apply_boundary_reassert, load_boundary_demos

# Constants
Change = Literal["escalation", "de-escalation", "no change", "change", "bad note"]
NonEscalationCategories = Literal[
    "continue_current_therapy",
    "lifestyle_changes",
    "patient_preference",
    "side_effects",
    "abnormal_labs",
    "defer_alternate_clinician",
    "defer_future_visit",
    "defer_after_labs",
    "ldl_close_to_goal",
    "ldl_at_goal",
    "recent_uptitration_of_therapy",
    "insurance_or_cost_issues",
    "other",
]
EscalationCategories = Literal[
    "ldl_above_goal",
    "risk_enhancing_factors",
    "ascvd_event",
    "rechallenge",
    "other",
]
TargetBinary = Literal["discussed", "not discussed"]
LDLTargets = Literal["<55", "<70", "<100", "None", "Other"]
LDLGoalStatus = Literal["at_goal", "close", "above", "unknown"]


class TargetPrompt(dspy.Signature):
    """You are a helpful and precise assistant for classifying whether an LDL target is discussed in the medical document provided, and if so, what the LDL target is. If mentioned, LDL target should fall under one of the following categories:
    <55
    <70
    <100
    None (goal/target is alluded to but not specified)
    Other (goal/target is not one of the numerical values above and not None) - if "Other", please specify the LDL target in the "other_target" field.

    Please provide your answer in the specified format.
    """

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField(desc="medical document")
    ldl_target_discussed: TargetBinary = dspy.OutputField(desc="whether LDL target was discussed or not")
    ldl_target_value: LDLTargets = dspy.OutputField(desc="specified LDL target in note")
    other_target: str = dspy.OutputField(desc="if LDL target is 'Other', specify the LDL target here")


class ChangePrompt(dspy.Signature):
    """You are a helpful and precise assistant for classifying whether there was a clinical adjustment in lipid-lowering therapy (LLT) based on the provided medical document.
    Your decision should be based on what happens at the current visit (even if prior visit actions are recorded in the note), which is typically found in the assessment/plan or impression/plan section of the note.
    Please provide your answer in the specified format.
    Options for clinical adjustment include:
    "escalation"   = increase dosing OR addition of new LLT on top of current therapy
    "de-escalation"= decrease dosing OR discontinuation/holding of LLT
    "change"       = switching from one LLT to another equivalent LLT (not purely add/remove)
    "no change"    = continuing current LLT without changes, continuing current dose of LLT (e.g. continue use of statin mentioned at existing dose)
    "bad note" = any note that is a brief addendum, nursing note, incomplete note, or anything besides a complete standard clinical note with history and assessment/plan from a provider
    """

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField(desc="medical document")
    llt_change: Change = dspy.OutputField(desc="whether there was a change in lipid-lowering therapy (LLT) or not")


class NonescalationReasons(dspy.Signature):
    """You are a helpful and precise assistant for classifying the reasons for non-escalation in lipid-lowering therapy (LLT) based on the provided medical document. Your goal is to minimize false positives in identifying reasons for non-escalation, and reasons should only be selected with clear and explicit evidence.
    Reason(s) for non-escalation should be based on what happens at the current visit (with the exception of patient-reported side effects), which are typically found in the assessment/plan or impression/plan section of the note.
    Please provide your answer in the specified format. Multiple categories can apply for the reasons, so if there are multiple reasons, list all that apply in the "llt_reasons" field.
    Please use all available information in the document, including the most recent LDL value if available, to determine whether the course of action was appropriate or not.
    All patients with defined ASCVD should be on LLT to target an LDL value no greater than 70. Unless a change in their LLT was very recent, the LLT should be escalated.
    Providers not following this guideline will either have a reason in the list below, have a reason not provided ("other"), or just plan to continue current therapy without clearly defining a reason why they aren't escalating therapy.
    If another reason is provided that does not fit into the specified categories, please classify as "other" and specify the reason in the "other_reasons" field. Note that almost every note should fall into one of the specified categories, so use "other" sparingly and only if there is truly no other category that fits the reason for non-escalation.

    LDL assessment (complete before selecting llt_reasons): identify the most recent numeric LDL value in the note and set recent_ldl_value (numeric string, or "none"). Then set ldl_goal_status: "at_goal" if LDL < 70; "close" if LDL 70-75; "above" if LDL > 75 or assumed > 75; "unknown" if no recent LDL value. Assume LDL is above goal (>75) unless a recent lab value shows otherwise.
    Critical LDL rules:
    - Do NOT select ldl_at_goal or ldl_close_to_goal from clinician language alone ("at goal", "well controlled", "lipids look good") when recent LDL is >75 or unknown — use continue_current_therapy.
    - A recent numeric LDL meeting the threshold is required for LDL goal labels.

    Two-stage reason selection:
    (1) Specific non-CCT reasons first — list every specific category with clear evidence.
    (2) Residual continue_current_therapy ONLY when stage (1) finds no qualifying specific reason. Never co-emit CCT with any specific reason.

    Defer_after_labs (STRICT — this is the most over-selected category; require explicit conditioning):
    - Select defer_after_labs ONLY when the assessment/plan explicitly conditions a FUTURE LLT change (titration up/down, adding or switching an agent, or reassessing the medication) on the result of a specific pending lab/imaging test. There must be language tying the lab result to an LLT decision, e.g. "lipid panel ordered, will titrate statin accordingly", "reassess need for atorvastatin 80 mg after labs", "if LDL still elevated on repeat panel, will add ezetimibe", "recheck lipids and adjust therapy accordingly", "will consider changing her statin pending LFTs".
    - Do NOT select defer_after_labs when the plan simply CONTINUES the current LLT and orders or repeats a lipid panel for routine surveillance/monitoring. Continuing therapy while monitoring lipids is continue_current_therapy, NOT defer_after_labs. Examples that are continue_current_therapy (NOT defer): "continue atorvastatin 80 mg, recheck lipids in 3-6 months", "cont statin - repeat lipid panel", "check labs with lipids in 6 months", "cont lipitor and recheck 3 mos", "continue current medication; lipid panel ordered".
    - A bare lipid panel order, screening labs, "future lab" order, or "labs today" without explicit LLT-decision language is continue_current_therapy.

    Options for non-escalation reasons include:
    "lifestyle_changes" = explicit trial of diet/exercise/supplements instead of LLT escalation
    "patient_preference" = explicitly documented patient preference to not modify LLT that are independent from reported side effects. If patient is non-adherent to prescribed LLT, "patient_preference" should be selected.
    "side_effects" = patient-reported side effects (myalgia, diarrhea, etc.). Select side_effects whenever the note documents LLT/statin intolerance or side effects anywhere — including Allergies, HPI, or plan — even if described as historical, from a prior agent, or not explicitly re-linked to the current visit's decision. Do NOT withhold side_effects just because the intolerance is documented in Allergies rather than the assessment/plan, or because it involves a previously tried statin. Only omit side_effects if the note explicitly refutes that the symptoms were LLT-related.
    "abnormal_labs" = e.g. abnormal CK, LFTs that would contraindicate LLT changes
    "defer_alternate_clinician" = deferring LLT modifications to another clinician. This should be explicitly mentioned in the assessment/impression and plan.
    "defer_future_visit" = deferring LLT modifications to a future clinic visit, with explicit rationale documented as to why this is being deferred. This should be explicitly mentioned in the assessment/impression and plan. Include conditional future LLT plans timed to a later visit (e.g., "If LDL remains >70 in 6 months, would favor adding ezetimibe"). Routine RTC alone is not enough.
    "defer_after_labs" = re-assessing LLT modifications pending lab/imaging results, where the plan explicitly conditions a future LLT change on those results (see the STRICT rule above). Routine "continue current statin + recheck/repeat lipids" surveillance is continue_current_therapy, not defer_after_labs. A "Future lab" order without explicit LLT-decision language is continue_current_therapy.
    "ldl_close_to_goal" = LDL between 70 and 75; select ONLY when ldl_goal_status is "close"
    "ldl_at_goal" = LDL < 70 with recent lab value; select ONLY when ldl_goal_status is "at_goal"
    "recent_uptitration_of_therapy" = recent escalation of LLT prior to visit
    "insurance_or_cost_issues" = documentation of insurance or cost issues that would limit LLT escalation
    "continue_current_therapy" = continuation of current therapy without a clearly acceptable reason (from the list above) provided for non-escalation of LLT. This includes cases where LLT is not directly discussed in the note as well (absence of discussion implies continuation of therapy). This also includes cases where the clinician assumes the LDL is at a goal value that is discordant with current guidelines (either they have the wrong target or the LDL is actually above target).
    "other" = a genuinely novel, explicitly documented rationale for non-escalation that fits NONE of the categories above. Be extremely conservative: "other" is almost always wrong. If the clinician simply continues therapy, uses a questionable/absent rationale, discontinues without a stated acceptable reason, or the note lacks a clear categorizable reason, use continue_current_therapy — NOT other. Do not use "other" as a catch-all for weak or implicit reasoning. Only use "other" when there is a specific stated reason that clearly does not map to any listed category (and always populate other_reasons).

    For every selected label in llt_reasons except continue_current_therapy, also provide reason_evidence: a dict mapping that label to one or more verbatim quote spans copied exactly from the note (literal substrings; do not paraphrase). Prefer quotes from the assessment/plan or impression/plan. For side_effects only, quotes may come from Allergies or HPI. continue_current_therapy needs no quote.
    """

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField(desc="medical document")
    recent_ldl_value: str = dspy.OutputField(desc="most recent LDL value as string, or 'none' if unavailable")
    ldl_goal_status: LDLGoalStatus = dspy.OutputField(
        desc="LDL goal status assessed from recent lab: at_goal (<70), close (70-75), above (>75), unknown"
    )
    llt_reasons: List[NonEscalationCategories] = dspy.OutputField(
        format=list, desc="categories from the allowed list"
    )
    other_reasons: dict[str, list[str]] = dspy.OutputField(
        desc="reasons other than those specified in the categories"
    )
    reason_evidence: dict[str, list[str]] = dspy.OutputField(
        desc="verbatim note quote spans per selected llt_reasons label (except continue_current_therapy)"
    )


class EscalationReasons(dspy.Signature):
    """You are a helpful and precise assistant for classifying the reasons for escalation in lipid-lowering therapy (LLT) based on the provided medical document.
    Prior visit actions may be recorded in the note, but the LLT change and reason should be based on what happens at the current visit, which is typically found in the assessment/plan or impression/plan section of the note.

    Please provide your answer in the specified format. Multiple categories can apply for the reasons, so if there are multiple reasons, list all that apply in the "llt_reasons" field.
    If an other reason is provided that does not fit into the specified categories, please classify as "other" and specify the reason in the "other_reasons" field.
    Options for escalation reasons include:
    "ldl_above_goal"
    "risk_enhancing_factors" (e.g., family history, active substance use)
    "ascvd_event" (e.g., significant myocardial infarction, stroke)
    "rechallenge" (e.g., old discontinuation / side effects are not evidenced)
    "other" = reason not specified above, provide details in "other_reasons" field
    """

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField(desc="medical document")
    llt_reasons: List[EscalationCategories] = dspy.OutputField(
        format=list, desc="categories from the allowed list"
    )
    other_reasons: dict[str, list[str]] = dspy.OutputField(
        desc="reasons other than those specified in the categories"
    )


def apply_nonescalation_gates(prediction, text=None, labs_verifier=None, labs_adder=None):
    out = apply_ldl_goal_gate(prediction)
    out = apply_evidence_span_gate(out, text=text)
    out = apply_defer_recall_add(out, text=text, adder=labs_adder)
    out = apply_defer_labs_verifier(out, text=text, verifier=labs_verifier)
    out = apply_other_suppression_gate(out)
    out = dict(out)
    out.pop("reason_evidence", None)
    return out


class PromptRouter_Predict(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route_logic = dspy.Predict(ChangePrompt)
        self.escalation = dspy.Predict(EscalationReasons)
        self.nonescalation = dspy.Predict(NonescalationReasons)
        self.labs_verifier = dspy.Predict(DeferAfterLabsVerify)

    def forward(self, context, text):
        preds = self.route_logic(context=context, text=text)
        llt_change = preds.llt_change
        if llt_change == "escalation":
            return dict(preds) | dict(self.escalation(context=context, text=text))
        elif llt_change in ["de-escalation", "no change", "change"]:
            out = dict(preds) | dict(self.nonescalation(context=context, text=text))
            return apply_nonescalation_gates(out, text=text, labs_verifier=self.labs_verifier)
        else:
            return dict(preds) | {"llt_reasons": [], "other_reasons": {}}


class PromptRouter_ChainOfThought(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route_logic = dspy.ChainOfThought(ChangePrompt)
        self.escalation = dspy.ChainOfThought(EscalationReasons)
        self.nonescalation = dspy.ChainOfThought(NonescalationReasons)
        self.labs_verifier = dspy.Predict(DeferAfterLabsVerify)
        self.labs_adder = dspy.Predict(DeferAfterLabsAdd)
        self.boundary = dspy.ChainOfThought(BoundaryReassert)
        self.boundary.demos = load_boundary_demos()

    def forward(self, context, text):
        preds = self.route_logic(context=context, text=text)
        llt_change = preds.llt_change
        if llt_change == "escalation":
            return dict(preds) | dict(self.escalation(context=context, text=text))
        elif llt_change in ["de-escalation", "no change", "change"]:
            out = dict(preds) | dict(self.nonescalation(context=context, text=text))
            out = apply_boundary_reassert(out, text=text, context=context, reasoner=self.boundary)
            return apply_nonescalation_gates(out, text=text, labs_verifier=self.labs_verifier, labs_adder=self.labs_adder)
        else:
            return dict(preds) | {"llt_reasons": [], "other_reasons": {}}


class PromptRouter_LDL_Target(dspy.Module):
    def __init__(self):
        super().__init__()
        self.target = dspy.ChainOfThought(TargetPrompt)

    def forward(self, context, text):
        preds = self.target(context=context, text=text)
        return dict(preds)
