"""
BiasGuard — Interactive Streamlit Demo
=======================================
A polished, production-quality frontend for the BiasGuard
bias detection system.

Features:
- Document upload (PDF, DOCX, TXT) or paste text directly
- Real-time analysis with animated progress
- Visual bias heatmap with highlighted spans
- Category breakdown with Plotly charts
- Before/After comparison view
- Downloadable JSON/Markdown/PDF reports
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Page Config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BiasGuard — Hiring Bias Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolve_api_base() -> str:
    """Resolve API base URL from Streamlit secrets or environment."""
    api_from_secrets = st.secrets.get("STREAMLIT_API_BASE_URL")
    api_from_env = os.getenv("STREAMLIT_API_BASE_URL")
    api_base = api_from_secrets or api_from_env or "http://localhost:8000"
    return api_base.rstrip("/")


API_BASE = _resolve_api_base()

SAMPLE_INPUTS = {
    "Biased Job Description": {
        "doc_type": "job_description",
        "text": """Senior Software Engineer — GrowthOS (Series B, $50M raised)

We're looking for a young, energetic rockstar developer who can hit the ground running 
and dominate our engineering challenges. Must be a culture fit with our tight-knit startup 
family. Native English speaker preferred.

Requirements:
• Recent graduate from a top/elite university (prefer Ivy League)
• Digital native comfortable with our cutting-edge stack
• Must be a tech-savvy ninja who thinks outside the box
• Work hard, play hard mentality — we move fast and break things
• Competitive and aggressive approach to problem-solving
• Looking for a hungry self-starter, not someone who's overqualified

Bonus points:
• Strong and dominant presence in open-source community
• Fresh perspective, not old-school in your thinking

To apply: Submit CV with photo to hiring@growthOS.com
Salary: Depends on experience (include salary history in application)""",
    },
    "Biased Interview Transcript": {
        "doc_type": "interview_transcript",
        "text": """Interview Transcript — Candidate: Jordan Smith
Position: Product Manager
Interviewer: Sarah L.

Sarah: Thanks for coming in, Jordan. Can you start by telling me about yourself?

Jordan: Sure, I have 8 years of experience in product management across fintech and healthcare.

Sarah: Great. How old are you exactly? We're a very dynamic, young team here.

Jordan: I'm 42.

Sarah: Ah, okay. Do you have children? We often work late into the evening and weekends.

Jordan: I have two kids, yes.

Sarah: And what does your husband do? Does he work long hours too?

Jordan: I'm not married.

Sarah: Okay. Where are you originally from? Your name sounds... interesting.

Jordan: I'm from Chicago, originally.

Sarah: No, I mean where are you really from?

Jordan: [pause] My family is from South Korea.

Sarah: I see. Are you a US citizen? We need someone who's, you know, fully committed here.

Jordan: I'm a permanent resident with full work authorization.

Sarah: Great. Last question — what medications are you currently taking? We have a high-stress environment.

Jordan: I don't think that's relevant to my qualifications for this role.""",
    },
    "Clean Job Description": {
        "doc_type": "job_description",
        "text": """Senior Software Engineer — TechForward (Remote-friendly)

We're seeking an experienced software engineer to join our distributed engineering team. 
We welcome applicants at all career stages who meet the technical requirements below.

Requirements:
• 5+ years of software engineering experience
• Proficiency in Python and one or more of: Go, Rust, TypeScript
• Experience with distributed systems and microservices architecture
• Strong written and verbal communication skills
• Ability to collaborate across time zones in a remote environment

We offer:
• Competitive compensation ($150K–$200K depending on experience)
• Remote-first culture with quarterly in-person gatherings
• Comprehensive health insurance covering medical, dental, and vision
• 20 days PTO plus local public holidays
• $2,500 annual learning & development budget

TechForward is an equal opportunity employer. We evaluate all candidates based 
on their qualifications and do not discriminate based on race, gender, age, religion, 
disability, national origin, or any other protected characteristic.

Accommodations are available for candidates with disabilities — please contact us.""",
    },
}

SEVERITY_COLORS = {
    "NONE": "#2ECC71",
    "LOW": "#F39C12",
    "MEDIUM": "#E67E22",
    "HIGH": "#E74C3C",
    "CRITICAL": "#8E44AD",
}

CATEGORY_EMOJIS = {
    "GENDER_BIAS": "⚥",
    "AGE_BIAS": "🎂",
    "RACIAL_ETHNIC_BIAS": "🌍",
    "DISABILITY_BIAS": "♿",
    "SOCIOECONOMIC_BIAS": "💰",
    "APPEARANCE_BIAS": "👀",
    "COGNITIVE_STYLE_BIAS": "🧠",
    "INTERVIEW_BIAS": "🚨",
}


# ─── CSS ───────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .bias-badge-HIGH { 
        background: #E74C3C; color: white; padding: 2px 8px; 
        border-radius: 4px; font-size: 0.75rem; font-weight: bold; 
    }
    .bias-badge-MEDIUM { 
        background: #E67E22; color: white; padding: 2px 8px; 
        border-radius: 4px; font-size: 0.75rem; font-weight: bold; 
    }
    .bias-badge-LOW { 
        background: #F39C12; color: white; padding: 2px 8px; 
        border-radius: 4px; font-size: 0.75rem; font-weight: bold; 
    }
    .bias-badge-CRITICAL {
        background: #8E44AD; color: white; padding: 2px 8px; 
        border-radius: 4px; font-size: 0.75rem; font-weight: bold;
        animation: pulse 1s infinite;
    }
    .score-gauge {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
    }
    .instance-card {
        border-left: 4px solid;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        background: rgba(255,255,255,0.05);
    }
    .highlighted-span {
        background: rgba(231, 76, 60, 0.3);
        border-bottom: 2px solid #E74C3C;
        padding: 1px 2px;
        border-radius: 2px;
    }
    .rewrite-box {
        background: rgba(46, 204, 113, 0.1);
        border-left: 3px solid #2ECC71;
        padding: 8px 12px;
        border-radius: 0 4px 4px 0;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── API Client ────────────────────────────────────────────────────────────

def call_analyze_api(
    text: str, doc_type: str, llm_provider: str | None = None
) -> dict | None:
    """Call the BiasGuard analysis API."""
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_BASE}/analyze",
                json={
                    "text": text,
                    "doc_type": doc_type,
                    "llm_provider": llm_provider,
                    "include_full_rewrite": True,
                },
            )
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        st.error(
            "⚠️ Cannot connect to BiasGuard API. "
            "Make sure the API server is running: `uvicorn api.main:app --reload`"
        )
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def check_api_health() -> bool:
    """Check if the API is reachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{API_BASE}/health")
            return r.status_code == 200
    except Exception:
        return False


# ─── Visualization Helpers ─────────────────────────────────────────────────

def render_score_gauge(score: float, severity: str) -> None:
    """Render a Plotly gauge chart for the overall bias score."""
    color = SEVERITY_COLORS.get(severity, "#E74C3C")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(score * 100, 1),
            title={"text": f"Bias Score<br><span style='font-size:0.8em;color:{color}'>{severity}</span>"},
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 15], "color": "#2ECC71"},
                    {"range": [15, 40], "color": "#F39C12"},
                    {"range": [40, 70], "color": "#E67E22"},
                    {"range": [70, 100], "color": "#E74C3C"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.75,
                    "value": score * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_category_chart(category_summary: dict) -> None:
    """Render a horizontal bar chart of bias by category."""
    if not category_summary:
        st.info("No bias categories to display.")
        return

    df = pd.DataFrame([
        {
            "Category": cat.replace("_", " ").replace("BIAS", "").strip(),
            "Count": data.get("count", 0),
            "HIGH": data.get("high", 0),
            "MEDIUM": data.get("medium", 0),
            "LOW": data.get("low", 0),
        }
        for cat, data in category_summary.items()
    ]).sort_values("Count", ascending=True)

    fig = go.Figure()
    for severity, color in [("HIGH", "#E74C3C"), ("MEDIUM", "#E67E22"), ("LOW", "#F39C12")]:
        fig.add_trace(go.Bar(
            name=severity,
            x=df[severity],
            y=df["Category"],
            orientation="h",
            marker_color=color,
        ))

    fig.update_layout(
        barmode="stack",
        title="Bias Instances by Category",
        height=max(250, len(df) * 50),
        margin=dict(l=0, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_bias_instance(instance: dict, idx: int) -> None:
    """Render a single bias instance as a styled card."""
    severity = instance.get("severity", "LOW")
    category = instance.get("category", "")
    color_map = {
        "HIGH": "#E74C3C",
        "MEDIUM": "#E67E22",
        "LOW": "#F39C12",
        "CRITICAL": "#8E44AD",
    }
    color = color_map.get(severity, "#F39C12")
    emoji = CATEGORY_EMOJIS.get(category, "⚠️")

    with st.container():
        st.markdown(
            f"""
            <div class="instance-card" style="border-left-color: {color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:bold; font-size:1.05rem;">
                        {emoji} &nbsp; "<mark style="background:rgba(231,76,60,0.3); padding:2px 4px; border-radius:3px;">{instance.get('span', '')}</mark>"
                    </span>
                    <span class="bias-badge-{severity}">{severity}</span>
                </div>
                <div style="margin-top:6px; font-size:0.85rem; opacity:0.8;">
                    <strong>Category:</strong> {category.replace('_', ' ')} &nbsp;|&nbsp;
                    <strong>Confidence:</strong> {round(instance.get('confidence', 0.8) * 100)}%
                </div>
                <div style="margin-top:8px; font-size:0.9rem;">
                    {instance.get('explanation', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if instance.get("rewrite_suggestion"):
            st.markdown(
                f"""
                <div class="rewrite-box" style="margin-top:6px;">
                    ✅ <strong>Suggested rewrite:</strong> "{instance['rewrite_suggestion']}"
                    <br><small>{instance.get('rewrite_explanation', '')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if instance.get("disadvantaged_groups"):
            groups = ", ".join(instance["disadvantaged_groups"])
            st.markdown(
                f"<div style='font-size:0.8rem;margin-top:4px;opacity:0.7;'>👥 Disadvantaged groups: {groups}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")


def generate_markdown_report(report: dict) -> str:
    """Generate a Markdown report from the analysis results."""
    severity = report.get("severity", "UNKNOWN")
    score = report.get("overall_bias_score", 0.0)
    instances = report.get("bias_instances", [])

    lines = [
        "# BiasGuard Analysis Report",
        "",
        f"**Run ID:** `{report.get('run_id', 'N/A')}`  ",
        f"**Document Type:** {report.get('doc_type', '').replace('_', ' ').title()}  ",
        f"**Overall Bias Score:** {round(score * 100, 1)}%  ",
        f"**Severity:** {severity}  ",
        f"**Bias Instances:** {len(instances)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        report.get("document_summary", ""),
        "",
    ]

    if report.get("most_critical_issues"):
        lines += ["## ⚠️ Most Critical Issues", ""]
        for issue in report["most_critical_issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    lines += ["## Bias Instances", ""]
    for i, inst in enumerate(instances, 1):
        lines += [
            f"### {i}. \"{inst.get('span', '')}\" — {inst.get('severity', '')}",
            f"**Category:** {inst.get('category', '').replace('_', ' ')}  ",
            f"**Explanation:** {inst.get('explanation', '')}",
            "",
        ]
        if inst.get("rewrite_suggestion"):
            lines += [
                f"> ✅ **Suggested rewrite:** *\"{inst['rewrite_suggestion']}\"*",
                f"> {inst.get('rewrite_explanation', '')}",
                "",
            ]

    if report.get("full_document_rewrite"):
        lines += [
            "---",
            "## Debiased Document",
            "",
            report["full_document_rewrite"],
        ]

    lines += [
        "",
        "---",
        "*Generated by BiasGuard — Bias-Detection RAG Agent for Hiring*",
        "*This report should be reviewed by qualified HR professionals.*",
    ]

    return "\n".join(lines)


# ─── Main App ──────────────────────────────────────────────────────────────

def main():
    inject_css()

    # Header
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size:2.5rem;">🛡️ BiasGuard</h1>
        <p style="font-size:1.1rem; opacity:0.7;">
            Bias-Detection RAG Agent for Hiring Documents
        </p>
    </div>
    """, unsafe_allow_html=True)

    # API Health Check
    api_online = check_api_health()
    if api_online:
        st.success("✅ API Connected", icon="✅")
    else:
        st.warning(
            "⚠️ API Offline — Using mock mode. "
            "For Streamlit Cloud, set `STREAMLIT_API_BASE_URL` in app Secrets. "
            "For local dev, start API with: `uvicorn api.main:app --reload`",
            icon="⚠️",
        )

    # ── Sidebar ────────────────────────────────────────────────────────────

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.caption(f"API Base: {API_BASE}")

        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "anthropic", "groq", "xai"],
            help="Select your LLM provider",
        )

        doc_type = st.selectbox(
            "Document Type",
            ["job_description", "resume", "interview_transcript"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        st.markdown("---")
        st.markdown("## 📚 Try an Example")
        example_name = st.selectbox("Load Example", list(SAMPLE_INPUTS.keys()))
        load_example = st.button("Load Example →", use_container_width=True)

        st.markdown("---")
        st.markdown("""
        ## 📊 About BiasGuard
        
        Detects **8 bias categories**:
        - ⚥ Gender Bias  
        - 🎂 Age Bias  
        - 🌍 Racial/Ethnic Bias  
        - ♿ Disability Bias  
        - 💰 Socioeconomic Bias  
        - 👀 Appearance Bias  
        - 🧠 Cognitive Style Bias  
        - 🚨 Interview Bias (prohibited questions)
        
        Powered by **LangGraph + ChromaDB + RAG**.  
        [GitHub](https://github.com/sameersemna/biasguard) · [Docs](https://biasguard.ai/docs)
        """)

    # ── Main Content ───────────────────────────────────────────────────────

    tab_analyze, tab_compare, tab_about = st.tabs(
        ["🔍 Analyze", "⚖️ Compare Before/After", "ℹ️ About"]
    )

    with tab_analyze:
        col_input, col_output = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### 📄 Input Document")

            if load_example:
                example = SAMPLE_INPUTS[example_name]
                st.session_state["input_text"] = example["text"]
                doc_type = example["doc_type"]

            input_text = st.text_area(
                "Paste document text",
                value=st.session_state.get("input_text", ""),
                height=400,
                placeholder="Paste your job description, resume, or interview transcript here...",
            )

            uploaded_file = st.file_uploader(
                "Or upload a file",
                type=["txt", "pdf", "docx"],
                help="PDF and DOCX extraction requires additional dependencies",
            )

            if uploaded_file and uploaded_file.type == "text/plain":
                input_text = uploaded_file.read().decode("utf-8")

            word_count = len(input_text.split()) if input_text else 0
            st.caption(f"📝 {word_count:,} words")

            analyze_btn = st.button(
                "🔍 Analyze for Bias",
                type="primary",
                use_container_width=True,
                disabled=not input_text.strip(),
            )

        with col_output:
            st.markdown("### 📊 Analysis Results")

            if analyze_btn and input_text.strip():
                with st.spinner("🤖 Running BiasGuard pipeline..."):
                    progress = st.progress(0, text="Retrieving bias patterns...")
                    time.sleep(0.3)
                    progress.progress(25, text="Analyzing document with LLM...")
                    time.sleep(0.3)

                    if api_online:
                        result = call_analyze_api(
                            text=input_text,
                            doc_type=doc_type,
                            llm_provider=llm_provider,
                        )
                    else:
                        # Mock response for demo without API
                        result = _mock_response(input_text, doc_type)

                    progress.progress(75, text="Generating rewrites...")
                    time.sleep(0.2)
                    progress.progress(100, text="Complete!")
                    time.sleep(0.3)
                    progress.empty()

                if result and result.get("success"):
                    report = result["report"]
                    st.session_state["last_report"] = report
                    st.session_state["last_input"] = input_text

                    _render_results(report)
                elif result:
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

            elif "last_report" in st.session_state:
                _render_results(st.session_state["last_report"])
            else:
                st.info(
                    "👈 Paste a document and click **Analyze for Bias** to get started.\n\n"
                    "Or load one of the example documents from the sidebar."
                )

    with tab_compare:
        if "last_report" in st.session_state and "last_input" in st.session_state:
            st.markdown("### ⚖️ Before vs After Debiasing")

            report = st.session_state["last_report"]
            original = st.session_state["last_input"]
            rewritten = report.get("full_document_rewrite", "No rewrite available")

            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("#### ❌ Original (with bias)")
                st.markdown(
                    f"<div style='background:rgba(231,76,60,0.05); padding:16px; border-radius:8px; "
                    f"border:1px solid rgba(231,76,60,0.3); min-height:300px; white-space:pre-wrap;'>{original}</div>",
                    unsafe_allow_html=True,
                )

            with col_after:
                st.markdown("#### ✅ Debiased Version")
                st.markdown(
                    f"<div style='background:rgba(46,204,113,0.05); padding:16px; border-radius:8px; "
                    f"border:1px solid rgba(46,204,113,0.3); min-height:300px; white-space:pre-wrap;'>{rewritten}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown(f"**Bias instances eliminated:** {report.get('bias_instance_count', 0)}")
        else:
            st.info("Run an analysis first to see the before/after comparison.")

    with tab_about:
        st.markdown("""
        ## About BiasGuard

        BiasGuard is an open-source, production-grade RAG system for detecting and mitigating 
        hiring bias in HR documents.

        ### Architecture
        The system runs a 4-agent LangGraph pipeline:

        1. **Retriever Agent** — Semantic search over a knowledge base of 200+ bias patterns
        2. **Analyzer Agent** — LLM-powered structured bias detection with span-level precision  
        3. **Mitigator Agent** — Context-aware neutral rewrite generation
        4. **Scorer Agent** — Weighted severity scoring with legal risk calibration

        ### Legal Framework
        BiasGuard's knowledge base is grounded in:
        - **Title VII** of the Civil Rights Act (1964) — race, sex, religion, national origin
        - **ADEA** — Age Discrimination in Employment Act (1967)
        - **ADA Title I** — Americans with Disabilities Act (1990)
        - **EEOC** pre-employment inquiry guidelines
        - Peer-reviewed hiring bias research

        ### Responsible Use
        This tool assists — it does not replace — human review. All findings should be 
        reviewed by qualified HR and legal professionals before action.

        ---
        **GitHub:** [github.com/sameersemna/biasguard](https://github.com/sameersemna/biasguard)  
        **License:** MIT
        """)


def _render_results(report: dict) -> None:
    """Render the analysis results panel."""
    score = report.get("overall_bias_score", 0.0)
    severity = report.get("severity", "UNKNOWN")
    instances = report.get("bias_instances", [])

    # Score gauge
    render_score_gauge(score, severity)

    # Quick stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Bias Instances", report.get("bias_instance_count", 0))
    c2.metric("Categories Detected", len(report.get("category_summary", {})))
    high_count = sum(
        1 for i in instances if i.get("severity") == "HIGH"
    )
    c3.metric("HIGH Severity", high_count, delta=None)

    # Category chart
    if report.get("category_summary"):
        render_category_chart(report["category_summary"])

    # Critical issues
    if report.get("most_critical_issues"):
        with st.expander("⚠️ Most Critical Issues", expanded=True):
            for issue in report["most_critical_issues"]:
                st.markdown(f"- {issue}")

    # Individual instances
    st.markdown(f"### 📋 Bias Instances ({len(instances)})")

    severity_filter = st.multiselect(
        "Filter by severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    )

    filtered = [i for i in instances if i.get("severity") in severity_filter]

    if not filtered:
        st.success("🎉 No bias detected matching the selected filters!")
    else:
        for idx, instance in enumerate(filtered):
            render_bias_instance(instance, idx)

    # Downloads
    st.markdown("### 📥 Download Report")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        json_str = json.dumps(report, indent=2)
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name=f"biasguard_report_{report.get('run_id', 'report')[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_dl2:
        md_report = generate_markdown_report(report)
        st.download_button(
            "⬇️ Download Markdown",
            data=md_report,
            file_name=f"biasguard_report_{report.get('run_id', 'report')[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _mock_response(text: str, doc_type: str) -> dict:
    """Return a mock response for demo purposes (when API is offline)."""
    import uuid

    has_bias = any(term in text.lower() for term in [
        "rockstar", "ninja", "young", "energetic", "culture fit",
        "native english", "recent graduate", "digital native", "how old"
    ])

    if not has_bias:
        return {
            "success": True,
            "report": {
                "run_id": str(uuid.uuid4()),
                "doc_type": doc_type,
                "overall_bias_score": 0.05,
                "severity": "NONE",
                "bias_instance_count": 0,
                "bias_instances": [],
                "category_summary": {},
                "document_summary": "No significant bias detected in this document. Well done!",
                "most_critical_issues": [],
                "full_document_rewrite": text,
                "performance": {
                    "total_duration_ms": 342.1,
                    "retrieval_duration_ms": 45.2,
                    "analysis_duration_ms": 210.3,
                    "mitigation_duration_ms": 68.1,
                    "scoring_duration_ms": 18.5,
                },
            },
        }

    return {
        "success": True,
        "report": {
            "run_id": str(uuid.uuid4()),
            "doc_type": doc_type,
            "overall_bias_score": 0.84,
            "severity": "HIGH",
            "bias_instance_count": 5,
            "bias_instances": [
                {
                    "id": str(uuid.uuid4()),
                    "span": "young, energetic",
                    "category": "AGE_BIAS",
                    "severity": "HIGH",
                    "explanation": "Directly excludes candidates over ~35. Violates ADEA.",
                    "disadvantaged_groups": ["Workers 40+", "Senior professionals"],
                    "rewrite_suggestion": "motivated, results-driven",
                    "rewrite_explanation": "Describes the desired quality without implying an age preference.",
                    "confidence": 0.97,
                },
                {
                    "id": str(uuid.uuid4()),
                    "span": "rockstar",
                    "category": "GENDER_BIAS",
                    "severity": "MEDIUM",
                    "explanation": "Male-coded term that reduces female applicant pools by up to 18% (Gaucher et al., 2011).",
                    "disadvantaged_groups": ["Women", "Non-binary candidates"],
                    "rewrite_suggestion": "exceptional engineer",
                    "rewrite_explanation": "Gender-neutral descriptor that conveys the same level of excellence.",
                    "confidence": 0.91,
                },
                {
                    "id": str(uuid.uuid4()),
                    "span": "culture fit",
                    "category": "RACIAL_ETHNIC_BIAS",
                    "severity": "HIGH",
                    "explanation": "Most-cited proxy for racial in-group preference. Highly correlated with homogeneity.",
                    "disadvantaged_groups": ["Racial minorities", "First-generation immigrants"],
                    "rewrite_suggestion": "values alignment (collaborative, accountable, transparent)",
                    "rewrite_explanation": "Specifying concrete values removes subjectivity and potential for bias.",
                    "confidence": 0.93,
                },
                {
                    "id": str(uuid.uuid4()),
                    "span": "native English speaker",
                    "category": "RACIAL_ETHNIC_BIAS",
                    "severity": "HIGH",
                    "explanation": "National origin discrimination. Illegal unless English fluency is a genuine job requirement.",
                    "disadvantaged_groups": ["Non-native English speakers", "Immigrants", "Bilingual candidates"],
                    "rewrite_suggestion": "fluent in English (written and verbal)",
                    "rewrite_explanation": "Fluency requirement is job-relevant; nativeness is not.",
                    "confidence": 0.98,
                },
                {
                    "id": str(uuid.uuid4()),
                    "span": "recent graduates",
                    "category": "AGE_BIAS",
                    "severity": "HIGH",
                    "explanation": "Effectively limits applicants to those in their 20s. Direct ADEA risk.",
                    "disadvantaged_groups": ["Experienced professionals 40+", "Career changers"],
                    "rewrite_suggestion": "candidates at all career stages",
                    "rewrite_explanation": "Opens the role to all qualified applicants regardless of career stage.",
                    "confidence": 0.95,
                },
            ],
            "category_summary": {
                "AGE_BIAS": {"count": 2, "high": 2, "medium": 0, "low": 0},
                "GENDER_BIAS": {"count": 1, "high": 0, "medium": 1, "low": 0},
                "RACIAL_ETHNIC_BIAS": {"count": 2, "high": 2, "medium": 0, "low": 0},
            },
            "document_summary": (
                "This job description contains multiple HIGH-severity bias indicators that "
                "create significant legal exposure and will substantially reduce applicant "
                "diversity. Immediate revision is recommended."
            ),
            "most_critical_issues": [
                "\"native English speaker\" — direct national origin discrimination, high EEOC risk",
                "\"culture fit\" — top proxy for racial in-group favoritism, must be replaced with specific criteria",
                "\"young, energetic\" + \"recent graduates\" — dual ADEA violations that will attract regulatory scrutiny",
            ],
            "full_document_rewrite": text.replace("young, energetic", "motivated, results-driven")
                .replace("rockstar", "exceptional engineer")
                .replace("culture fit", "values-aligned")
                .replace("native English speaker preferred", "fluent in English (written and verbal)")
                .replace("recent graduates", "candidates at all career stages"),
            "performance": {
                "total_duration_ms": 4821.3,
                "retrieval_duration_ms": 312.4,
                "analysis_duration_ms": 3201.7,
                "mitigation_duration_ms": 1102.8,
                "scoring_duration_ms": 204.4,
            },
        },
    }


if __name__ == "__main__":
    main()
