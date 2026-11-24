"""
Supabase Database Layer for Resume Scorer.
Handles CRUD operations for resumes, job descriptions, analysis results,
and Supabase Authentication (sign up, sign in, sign out, user session).
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get_client():
    """
    Create and return a Supabase client. Raises a clear error if keys are missing.
    """
    from supabase import create_client

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or "your-project-id" in url:
        raise ConnectionError(
            "SUPABASE_URL is not configured.\n"
            "→ Open .env and paste your Supabase Project URL.\n"
            "→ Find it at: Supabase Dashboard → Settings → API → Project URL"
        )
    if not key or key.startswith("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."):
        raise ConnectionError(
            "SUPABASE_KEY is not configured.\n"
            "→ Open .env and paste your anon/public API key.\n"
            "→ Find it at: Supabase Dashboard → Settings → API → anon public"
        )

    return create_client(url, key)


class ResumeDB:
    """Database and Authentication operations for the Resume Scorer system."""

    def __init__(self):
        self.client = _get_client()

    # ─── AUTHENTICATION ───────────────────────────────────────

    def sign_up(self, email: str, password: str) -> Dict[str, Any]:
        """Register a new user with Supabase Auth."""
        try:
            res = self.client.auth.sign_up({"email": email.strip(), "password": password})
            user = res.user
            session = res.session
            return {
                "success": True,
                "user": user,
                "session": session,
                "email": user.email if user else email,
                "user_id": user.id if user else None,
                "message": "Account created successfully! You are logged in." if session else "Account created! Please check your email to confirm or log in.",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sign_in(self, email: str, password: str) -> Dict[str, Any]:
        """Sign in an existing user with email and password."""
        try:
            res = self.client.auth.sign_in_with_password({"email": email.strip(), "password": password})
            user = res.user
            session = res.session
            return {
                "success": True,
                "user": user,
                "session": session,
                "email": user.email if user else email,
                "user_id": user.id if user else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def sign_out(self) -> bool:
        """Sign out the current user session."""
        try:
            self.client.auth.sign_out()
            return True
        except Exception:
            return False

    def get_current_user(self) -> Optional[Any]:
        """Fetch the authenticated user if session exists."""
        try:
            res = self.client.auth.get_user()
            return res.user if res else None
        except Exception:
            return None

    # ─── RESUMES ──────────────────────────────────────────────

    def save_resume(
        self,
        raw_text: str,
        candidate_name: Optional[str] = None,
        email: Optional[str] = None,
        file_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a resume and return the created row (with id)."""
        payload = {
            "raw_text": raw_text,
            "word_count": len(raw_text.split()),
        }
        if candidate_name:
            payload["candidate_name"] = candidate_name
        if email:
            payload["email"] = email
        if file_name:
            payload["file_name"] = file_name
        if user_id:
            payload["user_id"] = user_id

        try:
            result = self.client.table("resumes").insert(payload).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            # If user_id column doesn't exist yet in user's schema, retry without it
            if user_id and "user_id" in str(e).lower():
                payload.pop("user_id", None)
                result = self.client.table("resumes").insert(payload).execute()
                return result.data[0] if result.data else {}
            raise e

    def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single resume by ID."""
        result = (
            self.client.table("resumes")
            .select("*")
            .eq("id", resume_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def list_resumes(self, limit: int = 20, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent resumes, optionally filtered by user."""
        query = self.client.table("resumes").select(
            "id, candidate_name, email, file_name, word_count, created_at"
        )
        if user_id:
            try:
                query = query.eq("user_id", user_id)
            except Exception:
                pass
        result = query.order("created_at", desc=True).limit(limit).execute()
        return result.data or []

    # ─── JOB DESCRIPTIONS ────────────────────────────────────

    def save_job_description(
        self,
        raw_text: str,
        title: Optional[str] = None,
        company: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a job description and return the created row."""
        payload = {
            "raw_text": raw_text,
            "word_count": len(raw_text.split()),
        }
        if title:
            payload["title"] = title
        if company:
            payload["company"] = company
        if user_id:
            payload["user_id"] = user_id

        try:
            result = self.client.table("job_descriptions").insert(payload).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            if user_id and "user_id" in str(e).lower():
                payload.pop("user_id", None)
                result = self.client.table("job_descriptions").insert(payload).execute()
                return result.data[0] if result.data else {}
            raise e

    def get_job_description(self, jd_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single JD by ID."""
        result = (
            self.client.table("job_descriptions")
            .select("*")
            .eq("id", jd_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ─── ANALYSIS RESULTS ────────────────────────────────────

    def save_analysis(
        self,
        resume_id: str,
        jd_id: str,
        result_obj: Any,  # GapAnalysisResult pydantic model
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save a full analysis result linked to a resume and JD."""
        full_report = json.loads(result_obj.model_dump_json())

        payload = {
            "resume_id": resume_id,
            "jd_id": jd_id,
            "overall_score": result_obj.overall_score,
            "semantic_score": result_obj.semantic_score,
            "skills_score": result_obj.skills_score,
            "experience_score": result_obj.experience_score,
            "education_score": result_obj.education_score,
            "match_tier": result_obj.match_tier,
            "summary": result_obj.summary,
            "matched_skills": result_obj.matched_skills,
            "missing_skills": result_obj.missing_skills,
            "extra_skills": result_obj.extra_skills,
            "recommendations": result_obj.recommendations,
            "full_report": full_report,
        }
        if user_id:
            payload["user_id"] = user_id

        try:
            result = self.client.table("analysis_results").insert(payload).execute()
            return result.data[0] if result.data else {}
        except Exception as e:
            if user_id and "user_id" in str(e).lower():
                payload.pop("user_id", None)
                result = self.client.table("analysis_results").insert(payload).execute()
                return result.data[0] if result.data else {}
            raise e

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single analysis by ID."""
        result = (
            self.client.table("analysis_results")
            .select("*, resumes(candidate_name, file_name), job_descriptions(title, company)")
            .eq("id", analysis_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def list_analyses(self, limit: int = 20, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent analyses with resume and JD info, optionally filtered by user."""
        query = self.client.table("analysis_results").select(
            "id, overall_score, match_tier, summary, created_at, "
            "resumes(candidate_name, file_name), "
            "job_descriptions(title, company)"
        )
        if user_id:
            try:
                query = query.eq("user_id", user_id)
            except Exception:
                pass
        result = query.order("created_at", desc=True).limit(limit).execute()
        return result.data or []

    # ─── UTILITY ─────────────────────────────────────────────

    def health_check(self) -> bool:
        """Verify database connectivity."""
        try:
            self.client.table("resumes").select("id").limit(1).execute()
            return True
        except Exception:
            return False
