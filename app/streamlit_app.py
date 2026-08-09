"""
Smart Resume Analyser — Streamlit Application
==============================================

Main application entry point. Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Smart Resume Analyser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ────────────────────────────────────────────────────────
css_path = PROJECT_ROOT / "app" / "assets" / "style.css"
if css_path.is_file():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Cached Model Loading (@st.cache_resource) ─────────────────────────────

@st.cache_resource
def load_sentence_model():
    """Load sentence-transformer model (once per app session)."""
    from sentence_transformers import SentenceTransformer
    from src.config import SENTENCE_MODEL_NAME
    return SentenceTransformer(SENTENCE_MODEL_NAME)


@st.cache_resource
def load_spacy_model():
    """Load spaCy model (once per app session)."""
    import spacy
    from src.config import SPACY_MODEL_NAME
    try:
        return spacy.load(SPACY_MODEL_NAME)
    except OSError:
        from spacy.cli import download
        download(SPACY_MODEL_NAME)
        return spacy.load(SPACY_MODEL_NAME)


def _init_models():
    """Pre-warm models so they're ready when needed."""
    import src.matcher as matcher_mod
    import src.extractor as extractor_mod

    matcher_mod._model_cache = load_sentence_model()
    extractor_mod._nlp_cache = load_spacy_model()


# ── Import project modules (after path setup) ─────────────────────────────
from src.parser import parse_resume
from src.extractor import extract_all
from src.matcher import compute_skill_overlap
from src.scorer import score_resume, MatchResult
from src.recommender import recommend_jobs, JobRecommendation
from src.classifier import classify_resume
from src.llm_suggestions import is_llm_available, suggest_improvements, SuggestionResult
from src.report import generate_report
from app.components.charts import score_gauge, score_breakdown_bar, skill_comparison_chart


# ── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with upload controls."""
    with st.sidebar:
        st.markdown("## 📄 Smart Resume Analyser")
        st.markdown(
            '<p style="color: #8892b0; font-size: 0.85rem;">'
            "AI-powered resume analysis with semantic matching, "
            "ATS scoring, and job recommendations."
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Resume upload
        st.markdown("### 📤 Upload Resume")
        resume_file = st.file_uploader(
            "Drop your resume here",
            type=["pdf", "docx"],
            key="resume_upload",
            help="Supported formats: PDF, DOCX",
        )

        st.markdown("---")

        # Job description input
        st.markdown("### 📝 Job Description (Optional)")
        jd_input_method = st.radio(
            "Input method:",
            ["Paste text", "Upload file"],
            horizontal=True,
            key="jd_method",
        )

        jd_text = ""
        if jd_input_method == "Paste text":
            jd_text = st.text_area(
                "Paste job description here",
                height=180,
                placeholder="Paste full job description for resume-vs-JD matching... (or leave blank for resume-only ATS audit)",
                key="jd_textarea",
            )
        else:
            jd_file = st.file_uploader(
                "Upload JD file",
                type=["pdf", "docx", "txt"],
                key="jd_upload",
            )
            if jd_file is not None:
                if jd_file.name.endswith(".txt"):
                    jd_text = jd_file.read().decode("utf-8")
                else:
                    jd_doc = parse_resume(jd_file)
                    jd_text = jd_doc.raw_text

        st.markdown("---")

        # LLM toggle
        llm_enabled = False
        if is_llm_available():
            llm_enabled = st.checkbox(
                "🤖 Enable AI Suggestions",
                value=True,
                help="Uses OpenAI GPT-4o-mini for resume improvement tips",
            )
        else:
            st.info(
                "💡 Set `GROQ_API_KEY` in `.env` to enable AI-powered suggestions.",
                icon="🔑",
            )

        # Analyze button
        st.markdown("---")
        analyze_clicked = st.button(
            "🔍 Analyze Resume",
            use_container_width=True,
            type="primary",
        )

        return resume_file, jd_text, llm_enabled, analyze_clicked


# ── Dashboard Rendering ────────────────────────────────────────────────────

def render_scores(result: MatchResult):
    """Render score gauges and component breakdown."""
    st.markdown('<div class="section-header">📊 Score Summary</div>', unsafe_allow_html=True)

    if not result.has_jd:
        st.info(
            "📋 **Resume-Only Mode**: No Job Description provided. "
            "Overall score represents your resume's standalone ATS quality, formatting, and completeness.",
            icon="ℹ️",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                score_gauge(result.ats_score.total_score, "ATS Quality Score"),
                use_container_width=True,
                key="gauge_ats_only",
            )
        with col2:
            st.plotly_chart(
                score_gauge(result.ats_score.formatting_score, "Formatting Cleanliness"),
                use_container_width=True,
                key="gauge_fmt_only",
            )
        return

    # With JD: full 3 gauges + breakdown bar
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            score_gauge(result.overall_score, "Overall Match"),
            use_container_width=True,
            key="gauge_overall",
        )
    with col2:
        st.plotly_chart(
            score_gauge(result.ats_score.total_score, "ATS Score"),
            use_container_width=True,
            key="gauge_ats",
        )
    with col3:
        st.plotly_chart(
            score_gauge(result.skill_overlap_pct, "Skill Overlap"),
            use_container_width=True,
            key="gauge_skills",
        )

    labels = ["Semantic Match", "Skill Overlap", "Experience", "Education"]
    scores = [
        result.semantic_similarity,
        result.skill_overlap_pct,
        result.experience_match,
        result.education_match,
    ]
    st.plotly_chart(
        score_breakdown_bar(labels, scores, "Job Compatibility Breakdown"),
        use_container_width=True,
        key="breakdown_bar",
    )


def render_skill_analysis(result: MatchResult):
    """Render skill gap analysis without UI contradictions."""
    st.markdown('<div class="section-header">🎯 Skill Analysis</div>', unsafe_allow_html=True)

    gap = result.gap_analysis

    if not result.has_jd:
        st.markdown("**💡 Extracted Resume Skills** (from spaCy NER + ESCO Taxonomy)")
        if gap.extra_skills:
            tags = " ".join(
                f'<span class="skill-tag skill-matched">{s}</span>'
                for s in gap.extra_skills
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No explicit technical skills detected in the resume text.")
        return

    # With JD comparison
    st.plotly_chart(
        skill_comparison_chart(gap.matched_skills, gap.missing_skills, gap.extra_skills),
        use_container_width=True,
        key="skill_chart",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Matched Skills**")
        if gap.matched_skills:
            tags = " ".join(
                f'<span class="skill-tag skill-matched">{s}</span>'
                for s in gap.matched_skills
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No matching skills detected between resume and JD.")

    with col2:
        st.markdown("**❌ Missing Skills**")
        if gap.missing_skills:
            tags = " ".join(
                f'<span class="skill-tag skill-missing">{s}</span>'
                for s in gap.missing_skills
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.success("No critical skills missing for this role!", icon="🎉")

    if gap.missing_keywords:
        with st.expander("📝 High-Frequency JD Keywords Missing from Resume"):
            st.write(", ".join(gap.missing_keywords))


def render_ats_feedback(result: MatchResult):
    """Render ATS formatting feedback based on actual resume checks."""
    st.markdown('<div class="section-header">📋 ATS Formatting & Structure</div>', unsafe_allow_html=True)

    ats = result.ats_score

    # Sub-scores
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Sections", f"{ats.section_score:.0f}%")
    with col2:
        st.metric("Length", f"{ats.length_score:.0f}%")
    with col3:
        st.metric("Bullets", f"{ats.bullet_score:.0f}%")
    with col4:
        st.metric("Keywords/Verbs", f"{ats.keyword_density_score:.0f}%")
    with col5:
        st.metric("Contact Info", f"{ats.contact_score:.0f}%")

    if ats.feedback:
        for fb in ats.feedback:
            if "missing" in fb.lower() or "very" in fb.lower() or "excessive" in fb.lower():
                st.warning(fb, icon="⚠️")
            else:
                st.info(fb, icon="💡")


def render_recommendations(recommendations: list[JobRecommendation]):
    """Render job recommendations based on resume analysis."""
    st.markdown('<div class="section-header">💼 Job Recommendations</div>', unsafe_allow_html=True)

    if not recommendations:
        st.info("Job recommendations are currently unavailable.", icon="📂")
        return

    for i, rec in enumerate(recommendations[:8], 1):
        score_class = (
            "score-high" if rec.similarity_score >= 75
            else "score-medium" if rec.similarity_score >= 50
            else "score-low"
        )

        st.markdown(
            f"""
            <div class="rec-card">
                <div>
                    <span style="color: #8892b0; font-size: 0.85rem;">#{i}</span>
                    <span class="rec-role">{rec.role}</span>
                </div>
                <span class="rec-score {score_class}">{rec.similarity_score:.1f}% Match</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_classification(resume_text: str):
    """Render resume category classification."""
    result = classify_resume(resume_text)
    if result is None:
        return

    st.markdown('<div class="section-header">🏷️ Resume Category Classification</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(
            "Predicted Category",
            result.top_prediction.category,
            delta=f"{result.top_prediction.confidence:.1%} confidence",
        )

    with col2:
        if len(result.all_predictions) > 1:
            st.markdown("**Top Predicted Categories:**")
            for pred in result.all_predictions[:4]:
                pct = pred.confidence * 100
                st.progress(pred.confidence, text=f"{pred.category} — {pct:.1f}%")


def render_suggestions(suggestions: SuggestionResult):
    """Render LLM-powered suggestions in a clean card layout."""
    st.markdown('<div class="section-header">🤖 AI Improvement Suggestions (Powered by Groq)</div>', unsafe_allow_html=True)

    if suggestions.error:
        st.error(f"Error generating suggestions: {suggestions.error}", icon="❌")
        return

    if suggestions.summary:
        st.markdown(
            f"""
            <div class="ai-summary-box">
                <div style="font-weight: 700; color: #64ffda; margin-bottom: 0.3rem;">💡 Executive Coach Summary</div>
                <div style="color: #ccd6f6; font-size: 0.95rem;">{suggestions.summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not suggestions.suggestions:
        st.success("No specific issues found! Your resume aligns well.", icon="✨")
        return

    for i, s in enumerate(suggestions.suggestions, 1):
        sev = s.severity.lower()
        badge_cls = "badge-high" if sev == "high" else "badge-medium" if sev == "medium" else "badge-low"
        card_cls = "ai-card-high" if sev == "high" else "ai-card-medium" if sev == "medium" else "ai-card-low"

        st.markdown(
            f"""
            <div class="ai-card {card_cls}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                    <div>
                        <span class="badge-category">📌 {s.category}</span>
                        <span class="{badge_cls}">{s.severity} priority</span>
                    </div>
                    <span style="color: #8892b0; font-size: 0.8rem; font-weight: 600;">#{i}</span>
                </div>
                <div style="color: #ccd6f6; font-weight: 600; font-size: 1rem; margin-bottom: 0.5rem;">
                    {s.suggestion}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if s.original or s.explanation:
            with st.expander(f"🔍 Details & Implementation Guidance for #{i}"):
                if s.original:
                    st.markdown(f"**Original Resume Text:**")
                    st.code(s.original, language="markdown")
                if s.explanation:
                    st.markdown(f"**Why this matters:** {s.explanation}")


def render_extracted_info(result: MatchResult):
    """Render extracted contact/experience info."""
    info = result.resume_info
    if info is None:
        return

    with st.expander("📋 Extracted Resume Details & Parsing Diagnostics"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Contact Information**")
            st.write(f"👤 Name: {info.contact.name or 'Not detected'}")
            st.write(f"📧 Email: {info.contact.email or 'Not detected'}")
            st.write(f"📱 Phone: {info.contact.phone or 'Not detected'}")
            exp_str = f"~{info.experience_years:.0f} years" if info.experience_years and info.experience_years > 0 else "0 years / Not specified"
            st.write(f"⏳ Experience: {exp_str}")

        with col2:
            st.markdown("**Education & Roles**")
            if info.education:
                for edu in info.education[:3]:
                    st.write(f"🎓 {edu}")
            else:
                st.write("🎓 None detected")

            if info.job_titles:
                st.markdown("**Detected Job Titles**")
                for title in info.job_titles[:4]:
                    st.write(f"💼 {title}")

            st.markdown("**Sections Found**")
            st.write(", ".join(info.sections_found) if info.sections_found else "None detected")


# ── Main App Logic ─────────────────────────────────────────────────────────

def main():
    # Pre-warm AI models
    with st.spinner("Loading AI models..."):
        _init_models()

    # Sidebar inputs
    resume_file, jd_text, llm_enabled, analyze_clicked = render_sidebar()

    # Create session state cache for analysis results
    if "analysis_state" not in st.session_state:
        st.session_state["analysis_state"] = None

    # Handle analysis execution when user clicks "Analyze Resume" or when state exists
    cache_key = (
        getattr(resume_file, "name", None),
        getattr(resume_file, "size", None),
        hash(jd_text.strip()),
        llm_enabled,
    )

    if analyze_clicked:
        if resume_file is None:
            st.error("Please upload a resume file first.", icon="📄")
            return

        with st.spinner("Parsing resume text..."):
            resume_doc = parse_resume(resume_file)

        if resume_doc.is_empty:
            st.error(
                "Could not extract text from uploaded file. "
                + " ".join(resume_doc.warnings),
                icon="❌",
            )
            return

        resume_text = resume_doc.raw_text

        with st.spinner("Analyzing resume content, ATS scores, and skills..."):
            match_result = score_resume(resume_text, jd_text)

        with st.spinner("Finding job recommendations..."):
            recommendations = recommend_jobs(resume_text, top_n=8)

        suggestions = SuggestionResult()
        if llm_enabled:
            with st.spinner("Generating AI suggestions with GPT-4o-mini..."):
                suggestions = suggest_improvements(resume_text, jd_text)

        # Store in session state
        st.session_state["analysis_state"] = {
            "cache_key": cache_key,
            "resume_text": resume_text,
            "jd_text": jd_text,
            "match_result": match_result,
            "recommendations": recommendations,
            "suggestions": suggestions,
            "llm_enabled": llm_enabled,
        }

    # Retrieve current analysis state if available
    analysis_state = st.session_state.get("analysis_state")

    if not analysis_state:
        # Landing Page
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 1rem;">
                <h1 style="color: #ccd6f6; font-size: 2.5rem; font-weight: 700;">
                    📄 Smart Resume Analyser
                </h1>
                <p style="color: #8892b0; font-size: 1.1rem; max-width: 600px; margin: 1rem auto;">
                    Upload your resume and optional job description to receive an
                    AI-powered ATS score, semantic matching, skill gap analysis,
                    and personalized job recommendations.
                </p>
                <div style="margin-top: 1.5rem;">
                    <span style="color: #64ffda; font-size: 0.95rem; font-weight: 500;">
                        ← Upload a resume in the sidebar to get started
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        features = [
            ("🎯", "Semantic Matching", "AI-powered resume-to-JD comparison using sentence embeddings"),
            ("📊", "ATS Scoring", "Checks section coverage, bullet points, length, and format cleanliness"),
            ("🔍", "Gap Analysis", "Identifies missing skills and keywords with alias normalization"),
            ("💼", "Job Recommendations", "Ranked job role matches based on your candidate profile"),
        ]
        for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
            with col:
                st.markdown(
                    f"""
                    <div class="score-card" style="text-align: center; min-height: 180px;">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                        <h3 style="color: #ccd6f6; font-size: 0.95rem;">{title}</h3>
                        <p style="color: #8892b0; font-size: 0.8rem;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        return

    # Render Active Analysis Results
    match_result: MatchResult = analysis_state["match_result"]
    recommendations: list[JobRecommendation] = analysis_state["recommendations"]
    suggestions: SuggestionResult = analysis_state["suggestions"]
    resume_text: str = analysis_state["resume_text"]

    # 1. Scores Summary
    render_scores(match_result)
    st.markdown("---")

    # 2. Skill Analysis & ATS Feedback
    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_skill_analysis(match_result)
    with col_right:
        render_ats_feedback(match_result)
    st.markdown("---")

    # 3. Recommendations & Classification
    render_recommendations(recommendations)
    render_classification(resume_text)
    st.markdown("---")

    # 4. LLM Suggestions
    if analysis_state["llm_enabled"]:
        render_suggestions(suggestions)
        st.markdown("---")

    # 5. Extracted Info Diagnostics
    render_extracted_info(match_result)
    st.markdown("---")

    # 6. PDF Report Download
    st.markdown('<div class="section-header">📥 Download PDF Report</div>', unsafe_allow_html=True)

    candidate_name = (
        match_result.resume_info.contact.name
        if match_result.resume_info and match_result.resume_info.contact.name
        else "Candidate"
    )

    try:
        report_bytes = generate_report(
            match_result=match_result,
            recommendations=recommendations,
            suggestions=suggestions if analysis_state["llm_enabled"] else None,
            candidate_name=candidate_name,
        )

        st.download_button(
            label="📄 Download Full Analysis Report (PDF)",
            data=bytes(report_bytes),
            file_name=f"resume_analysis_{candidate_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="pdf_download_btn",
        )
    except Exception as e:
        st.error(f"Error generating PDF report: {e}", icon="❌")


if __name__ == "__main__":
    main()
