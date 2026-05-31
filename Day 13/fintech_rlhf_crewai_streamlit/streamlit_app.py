from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from fintech_rlhf.crew_factory import CREWAI_AVAILABLE, run_fintech_crew
from fintech_rlhf.feedback_store import FeedbackStore
from fintech_rlhf.schemas import FintechScenario

load_dotenv()

st.set_page_config(
    page_title="Fintech RLHF CrewAI Demo",
    page_icon="🧠",
    layout="wide",
)

APP_TITLE = "Fintech RLHF with CrewAI"
APP_SUBTITLE = "U.S.-focused decision support demo with human feedback and audit-ready notes."

db_path = os.getenv("FEEDBACK_DB_PATH", "./data/feedback.db")
store = FeedbackStore(db_path=db_path)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Scenario inputs")
    customer_segment = st.selectbox(
        "Customer segment",
        ["Prime consumer", "Near-prime consumer", "Thin-file applicant", "Small business owner", "Existing customer"],
        index=1,
    )
    product_type = st.selectbox(
        "Fintech product",
        ["Personal loan", "Credit card", "BNPL", "Small business cash advance", "Digital wallet overdraft"],
        index=0,
    )
    decision_use_case = st.selectbox(
        "Decision use case",
        ["New application", "Credit line increase", "Fraud step-up review", "Account reopening", "Manual override review"],
        index=0,
    )
    requested_amount = st.number_input("Requested amount (USD)", min_value=100.0, value=5000.0, step=100.0)
    annual_income = st.number_input("Annual income (USD)", min_value=1000.0, value=65000.0, step=1000.0)
    debt_to_income = st.slider("Debt-to-income (%)", min_value=0.0, max_value=80.0, value=34.0, step=1.0)
    credit_band = st.selectbox(
        "Credit band",
        ["Excellent", "Prime", "Fair", "Near-prime", "Subprime", "Thin file"],
        index=2,
    )
    key_risks = st.multiselect(
        "Key risks",
        ["Thin file", "Recent late payment", "Identity mismatch", "Fraud velocity", "Charge-off history", "High utilization"],
    )
    desired_tone = st.selectbox(
        "Response tone",
        ["Direct and formal", "Supportive and plain-spoken", "Compliance-first", "Customer-friendly"],
        index=2,
    )

    run_button = st.button("Run Crew", use_container_width=True)

scenario = FintechScenario(
    customer_segment=customer_segment,
    product_type=product_type,
    decision_use_case=decision_use_case,
    requested_amount=requested_amount,
    annual_income=annual_income,
    debt_to_income=debt_to_income,
    credit_band=credit_band,
    key_risks=key_risks,
    desired_tone=desired_tone,
)

if "last_output" not in st.session_state:
    st.session_state.last_output = None

if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = None

if run_button:
    policy_memory = store.build_policy_memory()
    with st.spinner("Running the CrewAI workflow..."):
        output = run_fintech_crew(scenario, policy_memory)
    st.session_state.last_output = output
    st.session_state.last_scenario = scenario

tab1, tab2, tab3 = st.tabs(["Decision workflow", "Feedback dashboard", "Policy brief"])

with tab1:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.subheader("Current case")
        st.code(scenario.to_prompt_block(), language="text")

        st.info(
            "The app first tries a CrewAI-backed run. If no LLM credentials are present, "
            "it falls back to a deterministic local policy engine so the demo still works."
        )

        st.write(f"**CrewAI available:** {'Yes' if CREWAI_AVAILABLE else 'No'}")
        st.write(f"**Stored reviewer feedback entries:** {store.count_feedback()}")

    with col_b:
        st.subheader("Result")
        output = st.session_state.last_output
        if output is None:
            st.warning("Run the crew to generate a decision draft.")
        else:
            st.metric("Confidence / score", output.confidence)
            st.write(f"**Recommendation:** {output.recommendation}")
            st.write(f"**Next action:** {output.next_action}")
            st.markdown("**Rationale**")
            st.write(output.rationale)

            st.markdown("**Compliance notes**")
            for note in output.compliance_notes:
                st.write(f"- {note}")

            st.markdown("**Customer explanation**")
            st.write(output.customer_explanation)

    st.divider()

    if st.session_state.last_output is not None and st.session_state.last_scenario is not None:
        st.subheader("Human feedback")
        with st.form("feedback_form", clear_on_submit=False):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rating = st.slider("Overall rating", 1, 5, 4)
            with c2:
                clarity = st.slider("Clarity", 1, 5, 4)
            with c3:
                fairness = st.slider("Fairness", 1, 5, 4)
            with c4:
                compliance = st.slider("Compliance", 1, 5, 5)

            comments = st.text_area(
                "Reviewer comment",
                placeholder="What should be improved in the next run?",
                height=120,
            )
            save = st.form_submit_button("Save feedback")

        if save:
            store.save_feedback(
                st.session_state.last_scenario,
                st.session_state.last_output,
                rating=rating,
                clarity=clarity,
                fairness=fairness,
                compliance=compliance,
                comments=comments.strip(),
            )
            st.success("Feedback saved. Re-run the crew to see the next improvement cycle.")

with tab2:
    st.subheader("Feedback dashboard")
    total = store.count_feedback()
    avg_rating = store.average_rating()

    k1, k2 = st.columns(2)
    k1.metric("Total feedback records", total)
    k2.metric("Average rating", f"{avg_rating:.2f}" if total else "N/A")

    dist = store.rating_distribution()
    st.bar_chart(dist)

    st.markdown("**Recent feedback**")
    rows = store.recent_rows(limit=10)
    if not rows:
        st.info("No feedback saved yet.")
    else:
        for row in rows:
            scenario_json = json.loads(row["scenario_json"])
            st.write(
                f"- {row['created_at']} | Rating {row['rating']}/5 | "
                f"Clarity {row['clarity']} | Fairness {row['fairness']} | Compliance {row['compliance']}"
            )
            st.caption(f"{scenario_json.get('product_type')} • {scenario_json.get('decision_use_case')} • {row['comments']}")

with tab3:
    st.subheader("U.S. policy brief for this demo")
    policy_path = Path("knowledge/us_fintech_policy_brief.md")
    if policy_path.exists():
        st.markdown(policy_path.read_text(encoding="utf-8"))
    else:
        st.info("Policy brief file not found.")

    st.markdown("**Why this is a good RLHF-style demo**")
    st.write(
        "The crew generates a draft, the human corrects it, and the next draft absorbs the reviewer memory. "
        "That mirrors the practical feedback loop used in many production decision-support systems."
    )

    st.markdown("**U.S. sources reflected in the design**")
    st.write("- CFPB adverse-action expectations")
    st.write("- Federal Reserve model risk management guidance")
    st.write("- FINRA AI and GenAI obligations")
