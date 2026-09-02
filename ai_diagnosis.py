"""
AI diagnosis layer for NetSage Lite.

This module is deliberately kept separate from the rule engine and the UI.
Its only job: take a case (symptom + topology note + command output + rule
findings), ask an LLM for a structured diagnosis, and make sure whatever
comes back is trustworthy enough to show a human.

Three things happen here that are easy to skip but matter a lot for an
AI-assisted troubleshooting tool:
  1. The model's reply is forced through a strict schema (pydantic). If it
     doesn't match, we don't show it to the user -- we retry instead.
  2. Every "supporting_evidence" entry the model cites gets checked against
     the text we actually sent it. If the model cited something that wasn't
     in the input, that citation gets dropped rather than trusted.
  3. Confidence gets pulled down whenever citations got dropped, or when the
     deterministic rule checker didn't independently flag a failure. The
     idea: the model shouldn't get to sound more sure of itself than the
     hard evidence supports.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, Field, ValidationError

PROMPT_PATH = Path(__file__).parent / "prompts" / "diagnosis_prompt.txt"
PROMPT_VERSION = "netsage-lite-v1"
MAX_ATTEMPTS = 3
CONFIDENCE_CEILING_WHEN_UNSUPPORTED = 55  # out of 100


class DiagnosisError(Exception):
    """Raised when the model's output can't be trusted, even after retries."""


# ---------------------------------------------------------------------------
# Schema the model's reply must match. Anything that doesn't fit this shape
# gets rejected and retried rather than shown to the student.
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    source: str
    quote: str
    why_it_matters: str


class NextCheck(BaseModel):
    command: str
    reason: str


class DiagnosisContract(BaseModel):
    likely_cause: str
    confidence: int = Field(ge=0, le=100)
    layer: str
    topic: str
    supporting_evidence: list[EvidenceItem]
    other_possibilities: list[str] = []
    suggested_next_check: NextCheck
    suggested_fix: list[str]
    how_to_confirm_fix: list[str]
    caveats: str = ""
    requires_human_review: Literal[True] = True


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(
    symptom: str,
    topology_note: str,
    command_output: list[dict],
    rule_notes: list[str],
) -> str:
    """Fill the prompt template with this case's specifics.

    command_output is a list of {"source": ..., "text": ...} dicts -- keeping
    the source label attached to each block is what lets us later verify the
    model's citations actually point somewhere real.
    """
    output_text = "\n\n".join(
        f"[{i}] {block['source']}\n{block['text']}"
        for i, block in enumerate(command_output, start=1)
    ) or "(none provided)"

    rule_text = "\n".join(f"- {note}" for note in rule_notes) or "(no rule findings)"

    # Note: we use targeted .replace() rather than str.format() here because
    # the template file is full of literal JSON braces (the schema and the
    # worked example) -- .format() would try to interpret those as
    # placeholders too and crash.
    template = load_prompt_template()
    return (
        template.replace("{symptom}", symptom)
        .replace("{topology_note}", topology_note or "(none provided)")
        .replace("{command_output}", output_text)
        .replace("{rule_notes}", rule_text)
    )


# ---------------------------------------------------------------------------
# Calling the model (any OpenAI-compatible chat completions endpoint)
# ---------------------------------------------------------------------------

def call_model(user_prompt: str) -> str:
    api_key = os.environ.get("AI_API_KEY")
    base_url = os.environ.get("AI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("AI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise DiagnosisError(
            "AI_API_KEY is not set. Add it to your environment before calling the AI layer."
        )

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Return only valid JSON. No commentary."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Turning the raw model reply into a trustworthy DiagnosisContract
# ---------------------------------------------------------------------------

def parse_json_reply(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    # Sometimes a model wraps JSON in stray text despite instructions --
    # pull out the first {...} block and try again before giving up.
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise DiagnosisError("Model reply did not contain any JSON.")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise DiagnosisError(f"Model reply had malformed JSON: {exc}") from exc


def drop_unsupported_evidence(
    contract: DiagnosisContract, known_sources: list[str]
) -> tuple[DiagnosisContract, int]:
    """Remove any cited evidence whose source label doesn't match anything
    we actually sent the model. Returns the cleaned contract plus a count of
    how many citations were thrown out (used later to discount confidence).
    """
    normalized_known = [s.lower() for s in known_sources]

    def is_real_source(label: str) -> bool:
        label = label.strip().lower()
        return any(label in known or known in label for known in normalized_known)

    kept = [e for e in contract.supporting_evidence if is_real_source(e.source)]
    dropped_count = len(contract.supporting_evidence) - len(kept)

    if not kept:
        raise DiagnosisError("Every citation the model gave pointed to a source we never provided.")

    return contract.model_copy(update={"supporting_evidence": kept}), dropped_count


def adjust_confidence(
    contract: DiagnosisContract, rule_check_found_a_failure: bool, dropped_citations: int
) -> DiagnosisContract:
    """Cap confidence when the diagnosis isn't independently backed up.

    We only trust a high-confidence answer when (a) none of its citations had
    to be thrown out, AND (b) the deterministic rule checker separately
    flagged a real failure. Otherwise the model is essentially guessing with
    good vocabulary, and the UI shouldn't present that as near-certain.
    """
    if dropped_citations > 0 or not rule_check_found_a_failure:
        capped = min(contract.confidence, CONFIDENCE_CEILING_WHEN_UNSUPPORTED)
        return contract.model_copy(update={"confidence": capped})
    return contract


def diagnose(
    symptom: str,
    topology_note: str,
    command_output: list[dict],
    rule_notes: list[str],
    rule_check_found_a_failure: bool,
) -> dict:
    """Full pipeline: build prompt -> call model -> validate -> ground -> adjust.
    Retries up to MAX_ATTEMPTS times if the model's reply doesn't hold up.
    """
    prompt = build_prompt(symptom, topology_note, command_output, rule_notes)
    known_sources = [block["source"] for block in command_output]

    errors: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            raw = call_model(prompt)
            data = parse_json_reply(raw)
            contract = DiagnosisContract(**data)
            contract, dropped = drop_unsupported_evidence(contract, known_sources)
            contract = adjust_confidence(contract, rule_check_found_a_failure, dropped)
        except (DiagnosisError, ValidationError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
            continue

        result = contract.model_dump()
        result["prompt_version"] = PROMPT_VERSION
        result["attempts_used"] = attempt
        result["raw_model_reply"] = raw
        return result

    raise DiagnosisError(
        f"Could not get a trustworthy diagnosis after {MAX_ATTEMPTS} tries: " + " | ".join(errors)
    )


if __name__ == "__main__":
    # Quick manual smoke test -- run `python ai_diagnosis.py` after setting
    # AI_API_KEY to try it against a made-up case.
    sample = diagnose(
        symptom="PC on VLAN 10 can't ping its default gateway.",
        topology_note="Access switch SW1 trunks up to router R1 (router-on-a-stick).",
        command_output=[
            {"source": "PC1> ipconfig", "text": "IP: 10.1.10.5  Gateway: 10.1.10.1"},
            {"source": "R1# show ip interface brief", "text": "GigabitEthernet0/0.10  unassigned  down  down"},
        ],
        rule_notes=["IF-DOWN: fail - subinterface Gi0/0.10 is administratively down"],
        rule_check_found_a_failure=True,
    )
    print(json.dumps(sample, indent=2))
