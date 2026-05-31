"""
app.py — Streamlit Frontend for AI Financial Advisor
═════════════════════════════════════════════════════

Run: streamlit run app.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

    :root {
        --primary: #1a3c5e;
        --accent: #d4a843;
        --bg: #f8f6f1;
        --card: #ffffff;
        --text: #2c2c2c;
        --success: #2d7a4f;
        --warning: #b85c00;
        --danger: #c0392b;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--primary); }

    .metric-card {
        background: var(--card);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 4px solid var(--accent);
        margin-bottom: 12px;
    }

    .metric-label { font-size: 12px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: var(--primary); margin-top: 4px; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    .delta-pos { color: var(--success); }
    .delta-neg { color: var(--danger); }

    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        margin: 8px 0;
    }
    .risk-conservative { background: #fdecea; color: #c0392b; }
    .risk-moderate { background: #fef9e7; color: #b85c00; }
    .risk-aggressive { background: #eafaf1; color: #2d7a4f; }

    .report-section {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 15px;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2a5580; }

    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0dcd6;
        font-size: 15px;
    }

    .header-bar {
        background: linear-gradient(135deg, #1a3c5e 0%, #2a5580 100%);
        color: white;
        padding: 32px 40px;
        border-radius: 16px;
        margin-bottom: 28px;
    }
    .header-bar h1 { color: white !important; margin: 0; font-size: 2.2rem; }
    .header-bar p { color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: 16px; }

    .sidebar-section { margin-bottom: 24px; }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "risk_data" not in st.session_state:
    st.session_state.risk_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Load Pre-computed Data (no LLM) ──────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_summary_and_risk():
    from agents.data_agent import build_data_context
    from agents.risk_agent import build_risk_context
    data_ctx, summary = build_data_context()
    risk_ctx, risk_data = build_risk_context(summary)
    return summary, risk_data, data_ctx, risk_ctx


# ── Chart Helpers ─────────────────────────────────────────────────────────────

def make_category_pie(category_breakdown: dict):
    labels = list(category_breakdown.keys())
    values = list(category_breakdown.values())
    colors = ["#1a3c5e", "#2a5580", "#3a6fa0", "#4a89c0", "#d4a843",
              "#e8c070", "#2d7a4f", "#3d9a6f", "#c0392b", "#e74c3c", "#8e44ad", "#27ae60"]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=colors[:len(labels)]),
        textinfo="label+percent",
        textfont=dict(size=12),
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_monthly_bar(monthly_summary: list):
    if not monthly_summary:
        return None
    df = pd.DataFrame(monthly_summary)
    fig = go.Figure()
    fig.add_bar(x=df["month"], y=df.get("income", [0]*len(df)), name="Income", marker_color="#1a3c5e")
    fig.add_bar(x=df["month"], y=df.get("expenses", [0]*len(df)), name="Expenses", marker_color="#d4a843")
    fig.add_scatter(x=df["month"], y=df.get("net", [0]*len(df)), mode="lines+markers",
                   name="Net Savings", line=dict(color="#2d7a4f", width=2))
    fig.update_layout(
        barmode="group",
        margin=dict(t=20, b=40, l=40, r=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(tickangle=-30, gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee", tickprefix="₹"),
    )
    return fig


def make_budget_gauge(actual_pct: float, target_pct: float, label: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=actual_pct,
        number={"suffix": "%", "font": {"size": 20}},
        title={"text": label, "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, target_pct * 1.5], "tickwidth": 1},
            "bar": {"color": "#1a3c5e" if actual_pct <= target_pct else "#c0392b"},
            "threshold": {"line": {"color": "#d4a843", "width": 3}, "value": target_pct},
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💼 AI Financial Advisor")
    st.markdown("---")

    st.markdown("**Navigation**")
    page = st.radio("", ["📊 Dashboard", "🤖 AI Analysis", "💬 Ask Advisor", "📈 Evaluation"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Data Source**")
    data_source = st.radio("", ["Sample Data (Rahul Sharma)", "Upload My CSV"], label_visibility="collapsed")

    if data_source == "Upload My CSV":
        uploaded_file = st.file_uploader("Upload transactions CSV", type="csv")
    else:
        uploaded_file = None

    st.markdown("---")
    st.markdown("**OpenAI Model**")
    model_choice = st.selectbox("", ["gpt-4o-mini", "gpt-4o"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<small style='color:#888'>AI Financial Advisor v1.0<br>"
        "Built with CrewAI · OpenAI · Streamlit<br>"
        "⚠️ Not real financial advice</small>",
        unsafe_allow_html=True,
    )


# ── Load Data ─────────────────────────────────────────────────────────────────

try:
    summary, risk_data, data_ctx, risk_ctx = load_summary_and_risk()
    st.session_state.summary = summary
    st.session_state.risk_data = risk_data
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}\n\nMake sure your .env file has OPENAI_API_KEY set.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if "Dashboard" in page:
    st.markdown("""
    <div class="header-bar">
        <h1>💼 Financial Dashboard</h1>
        <p>Rahul Sharma · Software Engineer, Bengaluru · 6 months overview</p>
    </div>
    """, unsafe_allow_html=True)

    s = summary
    r = risk_data

    # ── KPI Row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Income</div>
            <div class="metric-value">₹{s['total_income']:,.0f}</div>
            <div class="metric-delta delta-pos">↑ Jan–Jun 2024</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Expenses</div>
            <div class="metric-value">₹{s['total_expenses']:,.0f}</div>
            <div class="metric-delta delta-neg">↑ Monitor closely</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Net Savings</div>
            <div class="metric-value">₹{s['net_savings']:,.0f}</div>
            <div class="metric-delta delta-pos">↑ {s['savings_rate_pct']}% savings rate</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        risk_class = r['risk_profile'].lower()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Risk Profile</div>
            <div class="metric-value">{r['risk_emoji']} {r['risk_profile']}</div>
            <div class="metric-delta">Score: {r['risk_score']}/12</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("#### 📅 Monthly Income vs Expenses")
        monthly_fig = make_monthly_bar(s.get("monthly_summary", []))
        if monthly_fig:
            st.plotly_chart(monthly_fig, use_container_width=True)

    with col_r:
        st.markdown("#### 🥧 Spending by Category")
        pie_fig = make_category_pie(s.get("category_breakdown", {}))
        st.plotly_chart(pie_fig, use_container_width=True)

    # ── Budget Gauges ─────────────────────────────────────────────────────────
    st.markdown("#### 🎯 50/30/20 Budget Rule Compliance")
    from agents.analysis_agent import analyse_budget_compliance
    compliance = analyse_budget_compliance(s)

    gc1, gc2, gc3 = st.columns(3)
    for (bucket, data), col in zip(compliance.items(), [gc1, gc2, gc3]):
        with col:
            g = make_budget_gauge(data["actual_pct"], data["target_pct"], bucket.replace("_", " ").title())
            st.plotly_chart(g, use_container_width=True)
            status_color = "delta-pos" if data["actual_pct"] <= data["target_pct"] else "delta-neg"
            st.markdown(f"<center><span class='{status_color}'>{data['status']}</span></center>", unsafe_allow_html=True)

    # ── Top Categories Table ──────────────────────────────────────────────────
    st.markdown("#### 📋 Category Breakdown")
    cat_df = pd.DataFrame([
        {"Category": k.replace("_", " ").title(),
         "Amount (₹)": f"₹{v:,.0f}",
         "% of Income": f"{v/s['total_income']*100:.1f}%"}
        for k, v in s["category_breakdown"].items()
    ])
    st.dataframe(cat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif "AI Analysis" in page:
    st.markdown("""
    <div class="header-bar">
        <h1>🤖 AI Pipeline Analysis</h1>
        <p>Run the full multi-agent pipeline: Data → Analysis → Risk → Advisory → QA</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 This runs 5 AI agents in sequence. Expect 30–90 seconds depending on the model.")

    query = st.text_input(
        "What would you like the AI to analyse?",
        value="Give me a complete financial health analysis and investment recommendations.",
        help="This becomes the final QA agent's question after all upstream reports are generated.",
    )

    col_btn, col_feedback = st.columns([1, 2])
    with col_btn:
        run_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)
    with col_feedback:
        feedback = st.text_input("Feedback from previous run (optional)", placeholder="e.g., 'Be more specific about tax savings'")

    if run_btn:
        try:
            from crew import FinancialAdvisorCrew
            crew = FinancialAdvisorCrew()

            with st.spinner("🔄 Running AI pipeline... (Data → Analysis → Risk → Advisory → QA)"):
                progress = st.progress(0, text="Initialising agents...")
                time.sleep(0.5)
                progress.progress(10, "Data Agent: Ingesting transactions...")
                result = crew.run_with_feedback(query, feedback or None)
                progress.progress(100, "Complete!")

            st.session_state.pipeline_result = result

            if result.success:
                st.success(f"✅ Pipeline completed in {result.elapsed_seconds}s")
            else:
                st.error(f"❌ Pipeline error: {result.error}")

        except ImportError:
            st.error("CrewAI not installed. Run: `pip install -r requirements.txt`")
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Display Results ───────────────────────────────────────────────────────
    if st.session_state.pipeline_result:
        r = st.session_state.pipeline_result

        tabs = st.tabs(["💬 Final Answer", "📊 Data Report", "🔍 Analysis", "⚖️ Risk", "💡 Advisory"])

        with tabs[0]:
            st.markdown("### 💬 AI Response to Your Query")
            st.markdown(f'<div class="report-section">{r.qa_response}</div>', unsafe_allow_html=True)
            st.markdown(f"*Risk Profile: {r.risk_profile} | Generated in {r.elapsed_seconds}s*")

        with tabs[1]:
            st.markdown("### 📊 Data Agent Report")
            st.markdown(f'<div class="report-section">{r.data_report}</div>', unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("### 🔍 Analysis Agent Report")
            st.markdown(f'<div class="report-section">{r.analysis_report}</div>', unsafe_allow_html=True)

        with tabs[3]:
            st.markdown("### ⚖️ Risk Agent Report")
            st.markdown(f'<div class="report-section">{r.risk_report}</div>', unsafe_allow_html=True)

        with tabs[4]:
            st.markdown("### 💡 Advisory Agent Report")
            st.markdown(f'<div class="report-section">{r.advisory_report}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASK ADVISOR
# ══════════════════════════════════════════════════════════════════════════════

elif "Ask Advisor" in page:
    st.markdown("""
    <div class="header-bar">
        <h1>💬 Ask Your Financial Advisor</h1>
        <p>Ask any financial question — the AI draws on your full financial profile to answer</p>
    </div>
    """, unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💡 Try these questions:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    suggested_q = ""
    with col_q1:
        if st.button("🚗 Can I afford a car loan?"):
            suggested_q = "Can I afford a car loan of ₹8 lakh?"
    with col_q2:
        if st.button("🏖️ Budget for vacation?"):
            suggested_q = "How much can I realistically budget for a vacation this year?"
    with col_q3:
        if st.button("📈 Am I saving enough?"):
            suggested_q = "Am I saving enough for retirement at my current rate?"

    col_q4, col_q5, col_q6 = st.columns(3)
    with col_q4:
        if st.button("🍔 Reduce food spend?"):
            suggested_q = "How can I reduce my food and dining expenses?"
    with col_q5:
        if st.button("💰 Best investments?"):
            suggested_q = "What investment instruments suit my risk profile?"
    with col_q6:
        if st.button("🧾 Tax savings?"):
            suggested_q = "How can I maximise my tax savings this financial year?"

    user_input = st.text_input("Or type your own question:", value=suggested_q)

    if st.button("💬 Get Answer", use_container_width=False) and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            from crew import FinancialAdvisorCrew
            crew = FinancialAdvisorCrew()

            with st.spinner("🤔 Thinking..."):
                result = crew.run(user_input)

            answer = result.qa_response if result.success else f"Error: {result.error}"
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation")
        for msg in reversed(st.session_state.chat_history[-10:]):
            if msg["role"] == "user":
                st.markdown(f"**🧑 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 Advisor:** {msg['content']}")
            st.markdown("---")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif "Evaluation" in page:
    st.markdown("""
    <div class="header-bar">
        <h1>📈 Evaluation Dashboard</h1>
        <p>Measure agent accuracy against ground-truth evaluation dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Load eval dataset
    with open(config.USER_PROFILE_JSON) as f:
        profile_data = json.load(f)
    eval_dataset = profile_data.get("evaluation_dataset", [])

    st.markdown(f"**Evaluation Dataset:** {len(eval_dataset)} test cases loaded")
    st.markdown("---")

    # Show eval cases
    st.markdown("### 📋 Evaluation Test Cases")
    for case in eval_dataset:
        with st.expander(f"**{case['id']}** — {case['query']}"):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Query:**")
                st.info(case["query"])
                st.markdown("**Expected Answer Contains:**")
                for kw in case.get("expected_answer_contains", []):
                    st.markdown(f"- `{kw}`")
            with col_r:
                st.markdown("**Ground Truth:**")
                gt = case.get("ground_truth_recommendation") or case.get("ground_truth_value", "N/A")
                st.success(gt)
                if "expected_risk_profile" in case:
                    st.markdown(f"**Expected Risk Profile:** `{case['expected_risk_profile']}`")
                if "expected_category" in case:
                    st.markdown(f"**Category:** `{case['expected_category']}`")

    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics Framework")

    metrics = [
        {"Metric": "Factual Accuracy", "Description": "Does the answer contain correct ₹ amounts?", "Tool": "deepeval / custom assertions", "Target": ">90%"},
        {"Metric": "Keyword Coverage", "Description": "Are expected keywords present in response?", "Tool": "ROUGE / exact match", "Target": ">85%"},
        {"Metric": "Risk Profile Match", "Description": "Is risk profile correctly classified?", "Tool": "Exact match", "Target": "100%"},
        {"Metric": "Response Latency", "Description": "Time to generate full pipeline response", "Tool": "time.time()", "Target": "<90s"},
        {"Metric": "Answer Relevance", "Description": "LLM-as-judge: is response on-topic?", "Tool": "deepeval GEval", "Target": ">0.8 / 1.0"},
        {"Metric": "Faithfulness", "Description": "Does response contradict input data?", "Tool": "deepeval Faithfulness", "Target": ">0.9 / 1.0"},
    ]
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🏃 Run Evaluation")
    st.info("Run `pytest tests/test_evaluation.py -v` in the terminal to execute the full evaluation suite.")

    # Show sample eval results (simulated)
    st.markdown("### 📊 Sample Results (Simulated)")
    sample_results = pd.DataFrame([
        {"Test Case": "EVAL_001", "Metric": "Keyword Coverage", "Score": "92%", "Pass": "✅"},
        {"Test Case": "EVAL_002", "Metric": "Keyword Coverage", "Score": "88%", "Pass": "✅"},
        {"Test Case": "EVAL_003", "Metric": "Factual Accuracy", "Score": "95%", "Pass": "✅"},
        {"Test Case": "EVAL_004", "Metric": "Keyword Coverage", "Score": "85%", "Pass": "✅"},
        {"Test Case": "EVAL_005", "Metric": "Pattern Match", "Score": "90%", "Pass": "✅"},
    ])
    st.dataframe(sample_results, use_container_width=True, hide_index=True)
