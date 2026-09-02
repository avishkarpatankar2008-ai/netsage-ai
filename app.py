"""
NetSage Lite -- Streamlit front end.

One file, three views (picked from the sidebar): run a case, browse past
cases, and read about the safety model. Everything reads/writes through
db.py so a case survives a page refresh.
"""

from __future__ import annotations

import json

import streamlit as st

import db
import simple_rules
from ai_diagnosis import DiagnosisError, diagnose

st.set_page_config(page_title="NetSage Lite", page_icon="🛠", layout="wide")
db.init_db()

if "command_blocks" not in st.session_state:
    st.session_state.command_blocks = [{"source": "", "text": ""}]
if "last_rule_results" not in st.session_state:
    st.session_state.last_rule_results = None
if "last_diagnosis" not in st.session_state:
    st.session_state.last_diagnosis = None
if "last_case_id" not in st.session_state:
    st.session_state.last_case_id = None


def render_command_blocks() -> None:
    st.caption("Paste each show-command / ipconfig output as its own block, labeled with where it came from.")
    for i, block in enumerate(st.session_state.command_blocks):
        cols = st.columns([1, 3])
        block["source"] = cols[0].text_input(
            "Source", value=block["source"], key=f"source_{i}", placeholder="e.g. PC1> ipconfig"
        )
        block["text"] = cols[1].text_area(
            "Output", value=block["text"], key=f"text_{i}", height=90, label_visibility="collapsed"
        )
    button_cols = st.columns([1, 1, 6])
    if button_cols[0].button("+ Add block"):
        st.session_state.command_blocks.append({"source": "", "text": ""})
        st.rerun()
    if len(st.session_state.command_blocks) > 1 and button_cols[1].button("- Remove last"):
        st.session_state.command_blocks.pop()
        st.rerun()


def render_diagnosis(result: dict) -> None:
    left, right = st.columns([2, 1])
    with left:
        st.subheader(result["likely_cause"])
        st.write(f"**Layer:** {result['layer']}  ·  **Topic:** {result['topic']}")

        st.markdown("**Supporting evidence**")
        for item in result["supporting_evidence"]:
            st.markdown(f"- *{item['source']}* — \u201c{item['quote']}\u201d — {item['why_it_matters']}")

        if result.get("other_possibilities"):
            st.markdown("**Other possibilities considered**")
            for alt in result["other_possibilities"]:
                st.markdown(f"- {alt}")

        st.markdown("**Suggested next check**")
        st.markdown(f"`{result['suggested_next_check']['command']}` — {result['suggested_next_check']['reason']}")

        st.markdown("**Proposed fix — requires review, not applied automatically**")
        for step in result["suggested_fix"]:
            st.markdown(f"1. {step}")

        st.markdown("**How to confirm the fix worked**")
        for step in result["how_to_confirm_fix"]:
            st.markdown(f"- {step}")

        if result.get("caveats"):
            st.info(f"Caveat: {result['caveats']}")

    with right:
        st.metric("Confidence", f"{result['confidence']}/100")
        st.progress(result["confidence"] / 100)
        st.caption(f"Prompt version: {result['prompt_version']} · attempts used: {result['attempts_used']}")
        st.warning("Human review required before any change.")


def workspace_view() -> None:
    st.title("🛠 NetSage Lite — Workspace")

    title = st.text_input("Case title", placeholder="e.g. VLAN 20 host can't reach gateway")
    symptom = st.text_area("Symptom", placeholder="What's actually broken, from the user's point of view?")
    topology_note = st.text_area("Topology note (optional)", placeholder="Any relevant layout details.")
    render_command_blocks()

    st.divider()
    run_col, ai_col = st.columns(2)

    if run_col.button("Run rule check", type="secondary"):
        blocks = [b for b in st.session_state.command_blocks if b["source"] and b["text"]]
        st.session_state.last_rule_results = simple_rules.run_all(blocks)

    if st.session_state.last_rule_results is not None:
        st.markdown("### Rule check results")
        if not st.session_state.last_rule_results:
            st.caption("No rules matched the output provided yet — add more command blocks.")
        for r in st.session_state.last_rule_results:
            icon = "✅" if r["status"] == "pass" else "❌"
            st.markdown(f"{icon} **{r['rule_id']}** — {r['finding']}")

    if ai_col.button("Run AI diagnosis", type="primary"):
        blocks = [b for b in st.session_state.command_blocks if b["source"] and b["text"]]
        rule_results = st.session_state.last_rule_results or simple_rules.run_all(blocks)
        rule_notes = [f"{r['rule_id']}: {r['status']} — {r['finding']}" for r in rule_results]
        rule_found_failure = any(r["status"] == "fail" for r in rule_results)

        case_id = db.save_case(title or "(untitled case)", symptom, topology_note, blocks)
        st.session_state.last_case_id = case_id
        db.save_rule_results(case_id, rule_results)

        with st.spinner("Asking the model for a diagnosis..."):
            try:
                result = diagnose(symptom, topology_note, blocks, rule_notes, rule_found_failure)
                st.session_state.last_diagnosis = result
                diagnosis_id = db.save_diagnosis(case_id, result)
                st.session_state.last_diagnosis_id = diagnosis_id
            except DiagnosisError as exc:
                st.session_state.last_diagnosis = None
                st.error(f"Couldn't get a diagnosis: {exc}")

    if st.session_state.last_diagnosis is not None:
        st.divider()
        st.markdown("### AI diagnosis")
        render_diagnosis(st.session_state.last_diagnosis)

        st.divider()
        st.markdown("### Review this diagnosis")
        decision = st.radio("Decision", ["Accept", "Edit", "Reject"], horizontal=True)
        note = st.text_area("Note (required for Edit or Reject)")
        if st.button("Submit review"):
            if decision in ("Edit", "Reject") and not note.strip():
                st.error("A note is required when editing or rejecting a diagnosis.")
            else:
                db.save_review(
                    st.session_state.last_case_id,
                    st.session_state.get("last_diagnosis_id"),
                    decision.lower(),
                    note.strip(),
                )
                st.success(f"Review recorded: {decision}.")


def history_view() -> None:
    st.title("📋 Case history")
    cases = db.list_cases()
    if not cases:
        st.caption("No cases yet — run one from the Workspace tab.")
        return

    for case in cases:
        with st.expander(f"#{case['id']} — {case['title']}"):
            bundle = db.get_case_bundle(case["id"])
            st.write(f"**Symptom:** {case['symptom']}")
            if case["topology_note"]:
                st.write(f"**Topology note:** {case['topology_note']}")

            if bundle["rules"]:
                st.markdown("**Rule results:**")
                for r in bundle["rules"]:
                    icon = "✅" if r["status"] == "pass" else "❌"
                    st.markdown(f"{icon} {r['rule_id']} — {r['finding']}")

            if bundle["diagnosis"]:
                result = json.loads(bundle["diagnosis"]["result_json"])
                st.markdown(f"**AI diagnosis:** {result['likely_cause']} ({result['confidence']}/100 confidence)")

            if bundle["review"]:
                st.markdown(f"**Review:** {bundle['review']['decision']} — {bundle['review']['note'] or '(no note)'}")
            else:
                st.caption("Not yet reviewed.")


def about_view() -> None:
    st.title("About NetSage Lite")
    st.markdown(
        """
NetSage Lite is a study aid, not an autonomous network technician. It looks at
the evidence you paste in and suggests what might be wrong — it never
connects to a real device and never applies a configuration change itself.

**Why every diagnosis needs a human decision:**
Language models are good at sounding confident even when they're wrong. This
tool tries to counter that in three ways: it checks the model's citations
against the text you actually provided, it lowers the reported confidence
whenever those citations don't hold up, and it refuses to treat any answer
as final until a person explicitly accepts, edits, or rejects it.

If you're using this for coursework, treat the AI's suggestion as a second
opinion to check your own reasoning against — not a substitute for it.
"""
    )


PAGES = {"Workspace": workspace_view, "Case History": history_view, "About": about_view}

st.sidebar.title("NetSage Lite")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
