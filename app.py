"""
Streamlit Web UI for BERT-Powered Resume Scorer & Gap Analyzer.
Features Supabase Authentication (Login, Sign-Up, Sign-Out) and User History Tracking.
Run:  streamlit run app.py
"""

import streamlit as st
import json
import tempfile
from pathlib import Path
from typing import Optional

from analyzer.scorer import ResumeScorer
from analyzer.database import ResumeDB


st.set_page_config(
    page_title="Resume Scorer & Gap Analyzer",
    page_icon="🎯",
    layout="wide",
)


@st.cache_resource
def load_scorer():
    return ResumeScorer()


def get_db():
    try:
        return ResumeDB()
    except Exception:
        return None


def init_session_state():
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "auth_token" not in st.session_state:
        st.session_state["auth_token"] = None


def render_score_bar(label: str, score: float):
    """Render a colored progress bar with label."""
    if score >= 80:
        color = "#22c55e"
    elif score >= 65:
        color = "#eab308"
    elif score >= 50:
        color = "#f97316"
    else:
        color = "#ef4444"

    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
            <span style="font-weight: 600;">{label}</span>
            <span style="font-weight: 700; color: {color};">{score}%</span>
        </div>
        <div style="background: #e5e7eb; border-radius: 8px; height: 14px; overflow: hidden;">
            <div style="width: {score}%; background: {color}; height: 100%; border-radius: 8px; transition: width 0.6s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_auth_sidebar(db):
    """Render Supabase authentication card in the sidebar."""
    st.sidebar.markdown("### 🔐 User Account")

    user = st.session_state.get("user")

    if user:
        st.sidebar.success(f"👤 Logged in as:\n**{user.get('email', 'User')}**")
        if st.sidebar.button("🚪 Log Out", use_container_width=True):
            if db:
                db.sign_out()
            st.session_state["user"] = None
            st.session_state["auth_token"] = None
            st.rerun()
    else:
        auth_tab = st.sidebar.radio("Account Options", ["Log In", "Sign Up"], horizontal=True)

        if auth_tab == "Log In":
            login_email = st.sidebar.text_input("Email", key="login_email")
            login_password = st.sidebar.text_input("Password", type="password", key="login_pwd")
            if st.sidebar.button("👉 Log In", type="primary", use_container_width=True):
                if not login_email or not login_password:
                    st.sidebar.error("Please enter both email and password.")
                elif not db:
                    st.sidebar.error("Supabase is not configured.")
                else:
                    with st.sidebar.status("Logging in..."):
                        res = db.sign_in(login_email, login_password)
                    if res.get("success"):
                        st.session_state["user"] = {
                            "email": res.get("email"),
                            "id": res.get("user_id"),
                        }
                        st.sidebar.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error(f"Login failed: {res.get('error', 'Invalid credentials')}")

        else:  # Sign Up
            signup_email = st.sidebar.text_input("Email", key="signup_email")
            signup_password = st.sidebar.text_input("Password (min 6 chars)", type="password", key="signup_pwd")
            if st.sidebar.button("✨ Create Account", type="primary", use_container_width=True):
                if not signup_email or not signup_password:
                    st.sidebar.error("Please fill in email and password.")
                elif len(signup_password) < 6:
                    st.sidebar.error("Password must be at least 6 characters.")
                elif not db:
                    st.sidebar.error("Supabase is not configured.")
                else:
                    with st.sidebar.status("Creating account..."):
                        res = db.sign_up(signup_email, signup_password)
                    if res.get("success"):
                        if res.get("user_id"):
                            st.session_state["user"] = {
                                "email": res.get("email"),
                                "id": res.get("user_id"),
                            }
                        st.sidebar.success(res.get("message", "Account created successfully!"))
                        st.rerun()
                    else:
                        st.sidebar.error(f"Sign up error: {res.get('error')}")

        st.sidebar.caption("💡 *You can also analyze resumes as a guest without logging in.*")


def render_history_tab(db):
    """Render past evaluations history tab."""
    st.markdown("### 📜 Past Analysis History")

    if not db:
        st.warning("Supabase database is not connected. Configure your `.env` file to view history.")
        return

    user = st.session_state.get("user")
    user_id = user.get("id") if user else None

    col_filter, col_refresh = st.columns([4, 1])
    with col_filter:
        if user:
            st.info(f"Showing history for: **{user.get('email')}**")
        else:
            st.caption("Showing global recent analyses. Log in to filter specifically by your account.")
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        analyses = db.list_analyses(limit=30, user_id=user_id)
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return

    if not analyses:
        st.info("No past resume analyses found yet. Run an analysis above to see it recorded here!")
        return

    for item in analyses:
        score = item.get("overall_score", 0)
        tier = item.get("match_tier", "Evaluation")
        date_str = str(item.get("created_at", ""))[:16].replace("T", " ")

        resume_info = item.get("resumes") or {}
        jd_info = item.get("job_descriptions") or {}
        candidate = resume_info.get("candidate_name") or resume_info.get("file_name") or "Unnamed Candidate"
        job_title = jd_info.get("title") or jd_info.get("company") or "Role Specification"

        # Badge color
        if score >= 80:
            badge = "🟢"
        elif score >= 65:
            badge = "🟡"
        elif score >= 50:
            badge = "🟠"
        else:
            badge = "🔴"

        with st.expander(f"{badge} **{candidate}** vs **{job_title}** — **{score}%** ({tier})  •  *{date_str}*"):
            st.markdown(f"**Executive Summary:** {item.get('summary', 'No summary available.')}")
            st.markdown(f"**Analysis ID:** `{item.get('id')}`")


def main():
    init_session_state()
    db = get_db()

    # Sidebar Authentication
    render_auth_sidebar(db)

    st.markdown("""
    <div style="text-align: center; padding: 15px 0 5px;">
        <h1 style="margin-bottom: 0;">🎯 Resume Scorer & Gap Analyzer</h1>
        <p style="color: #6b7280; font-size: 1.05rem;">
            BERT-powered semantic analysis · Skill taxonomy matching · ATS alignment scoring · Supabase Database
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main navigation tabs
    tab_analyze, tab_history = st.tabs(["🚀 Analyze Resume", "📜 My History"])

    with tab_history:
        render_history_tab(db)

    with tab_analyze:
        col_resume, col_jd = st.columns(2)

        with col_resume:
            st.subheader("📄 Resume")
            resume_upload = st.file_uploader(
                "Upload your resume (PDF or TXT)", type=["pdf", "txt"], key="resume_file"
            )
            resume_text = st.text_area(
                "Or paste your resume text below:",
                height=220,
                key="resume_text",
                placeholder="Paste your full resume text here...",
            )

        with col_jd:
            st.subheader("📋 Job Description")
            jd_upload = st.file_uploader(
                "Upload job description (TXT)", type=["txt"], key="jd_file"
            )
            jd_text = st.text_area(
                "Or paste the job description below:",
                height=220,
                key="jd_text",
                placeholder="Paste the full job description here...",
            )

        st.divider()

        analyze_btn = st.button("🚀 Analyze Resume & Detect Gaps", type="primary", use_container_width=True)

        if analyze_btn:
            # Resolve resume input
            resume_input = None
            resume_filename = None
            if resume_upload is not None:
                resume_filename = resume_upload.name
                suffix = Path(resume_upload.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(resume_upload.read())
                    resume_input = tmp.name
            elif resume_text.strip():
                resume_input = resume_text.strip()

            # Resolve JD input
            jd_input = None
            if jd_upload is not None:
                jd_input = jd_upload.read().decode("utf-8", errors="ignore")
            elif jd_text.strip():
                jd_input = jd_text.strip()

            if not resume_input:
                st.error("Please upload a resume or paste resume text.")
                return
            if not jd_input:
                st.error("Please upload a job description or paste JD text.")
                return

            with st.spinner("🔄 Loading BERT model and computing semantic embeddings..."):
                scorer = load_scorer()
                result = scorer.analyze(resume_input, jd_input)

            # Auto-save to Supabase
            user = st.session_state.get("user")
            user_id = user.get("id") if user else None

            if db:
                try:
                    raw_resume = scorer.resume_parser.extract_text(resume_input)
                    r_row = db.save_resume(
                        raw_text=raw_resume,
                        file_name=resume_filename,
                        user_id=user_id,
                    )
                    j_row = db.save_job_description(
                        raw_text=jd_input,
                        user_id=user_id,
                    )
                    if r_row.get("id") and j_row.get("id"):
                        db.save_analysis(
                            resume_id=r_row["id"],
                            jd_id=j_row["id"],
                            result_obj=result,
                            user_id=user_id,
                        )
                        st.toast("✅ Analysis saved to your Supabase history!", icon="💾")
                except Exception as e:
                    # Non-blocking if save fails
                    pass

            st.divider()

            # ━━━ OVERALL SCORE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            score = result.overall_score
            if score >= 80:
                score_color = "#22c55e"
                score_emoji = "🟢"
            elif score >= 65:
                score_color = "#eab308"
                score_emoji = "🟡"
            elif score >= 50:
                score_color = "#f97316"
                score_emoji = "🟠"
            else:
                score_color = "#ef4444"
                score_emoji = "🔴"

            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border: 2px solid {score_color}; border-radius: 16px; margin: 12px 0;">
                <div style="font-size: 3.2rem; font-weight: 800; color: {score_color};">{score_emoji} {score}%</div>
                <div style="font-size: 1.25rem; font-weight: 600; margin-top: 4px;">{result.match_tier}</div>
                <p style="color: #6b7280; margin-top: 6px; font-size: 0.95rem;">{result.summary}</p>
            </div>
            """, unsafe_allow_html=True)

            # ━━━ SUB-SCORES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.subheader("📊 Score Breakdown")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Semantic Alignment", f"{result.semantic_score}%")
            with c2:
                st.metric("Skills Match", f"{result.skills_score}%")
            with c3:
                st.metric("Experience", f"{result.experience_score}%")
            with c4:
                st.metric("Education", f"{result.education_score}%")

            render_score_bar("Semantic Alignment", result.semantic_score)
            render_score_bar("Skills Match", result.skills_score)
            render_score_bar("Experience", result.experience_score)
            render_score_bar("Education", result.education_score)

            # ━━━ SKILLS ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.divider()
            st.subheader("🔑 Skills Analysis")

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("**✅ Matched Skills**")
                if result.matched_skills:
                    for s in result.matched_skills:
                        st.markdown(f"<span style='background:#dcfce7;padding:3px 9px;border-radius:5px;margin:2px;display:inline-block;font-size:0.88rem;'>{s}</span>", unsafe_allow_html=True)
                else:
                    st.info("No matched skills detected")

            with sc2:
                st.markdown("**❌ Missing Skills (Gaps)**")
                if result.missing_skills:
                    for s in result.missing_skills:
                        st.markdown(f"<span style='background:#fee2e2;padding:3px 9px;border-radius:5px;margin:2px;display:inline-block;font-size:0.88rem;'>{s}</span>", unsafe_allow_html=True)
                else:
                    st.success("No missing skills!")

            with sc3:
                st.markdown("**➕ Extra Candidate Skills**")
                if result.extra_skills:
                    for s in result.extra_skills:
                        st.markdown(f"<span style='background:#dbeafe;padding:3px 9px;border-radius:5px;margin:2px;display:inline-block;font-size:0.88rem;'>{s}</span>", unsafe_allow_html=True)
                else:
                    st.info("No extra skills detected")

            # Category Breakdown
            if result.skills_by_category:
                st.markdown("#### 📂 Skills by Category")
                cat_data = []
                for cat, stats in result.skills_by_category.items():
                    cat_data.append({
                        "Category": cat.replace("_", " ").title(),
                        "Required in JD": stats.total_jd,
                        "Matched": stats.matched,
                        "Missing": stats.missing,
                        "Missing Skills": ", ".join(stats.missing_skills) if stats.missing_skills else "—",
                    })
                st.dataframe(cat_data, use_container_width=True, hide_index=True)

            # ━━━ REQUIREMENTS ALIGNMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if result.requirement_matches:
                st.divider()
                st.subheader(
                    f"🎯 JD Requirements Alignment  "
                    f"({result.strong_matches_count} ✅ | {result.moderate_matches_count} 🟡 | {result.gaps_count} ❌)"
                )

                for m in result.requirement_matches:
                    icon = {"strong_match": "✅", "moderate_match": "🟡", "gap": "❌"}.get(m.status, "❓")
                    with st.expander(f"{icon} {m.requirement[:95]}  —  **{m.similarity_score}%**"):
                        st.markdown(f"**Best Match in Resume:** {m.best_matching_text}")
                        if m.section_source:
                            st.caption(f"Found in section: `{m.section_source}`")

            # ━━━ EXPERIENCE & EDUCATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.divider()
            st.subheader("🏢 Experience & 🎓 Education")

            e1, e2 = st.columns(2)
            with e1:
                icon_map = {"meets": "✅", "exceeds": "⭐", "underqualified": "⚠️", "undetermined": "❓"}
                st.markdown(f"**Experience:** {icon_map.get(result.experience.status, '❓')} {result.experience.status.title()}")
                st.write(result.experience.details)
            with e2:
                st.markdown(f"**Education:** {icon_map.get(result.education.status, '❓')} {result.education.status.title()}")
                st.write(result.education.details)

            # ━━━ SECTION RELEVANCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if result.sections:
                st.divider()
                st.subheader("📑 Resume Section Relevance")
                sec_data = []
                for sec in result.sections:
                    sec_data.append({
                        "Section": sec.name,
                        "Present": "✅" if sec.present else "❌",
                        "Word Count": sec.word_count,
                        "JD Similarity": f"{sec.similarity_to_jd}%" if sec.present else "—",
                    })
                st.dataframe(sec_data, use_container_width=True, hide_index=True)

            # ━━━ RECOMMENDATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if result.recommendations:
                st.divider()
                st.subheader("💡 Actionable Recommendations")
                for r in result.recommendations:
                    st.markdown(f"- {r}")

            # ━━━ BULLET SUGGESTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if result.bullet_improvement_suggestions:
                st.divider()
                st.subheader("✏️ Bullet Point Enhancement Tips")
                for i, sug in enumerate(result.bullet_improvement_suggestions, 1):
                    st.markdown(f"**{i}. Original:** {sug.get('original', '')[:100]}")
                    st.info(f"💡 {sug.get('tip', '')}")

            # ━━━ ATS TIPS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if result.ats_formatting_tips:
                st.divider()
                st.subheader("📐 ATS Formatting Tips")
                for tip in result.ats_formatting_tips:
                    st.markdown(f"- {tip}")

            # ━━━ JSON DOWNLOAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            st.divider()
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=result.model_dump_json(indent=2),
                file_name="resume_gap_analysis_report.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
