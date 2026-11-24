"""
FastAPI REST API Server for Resume Scorer & Gap Analyzer.
Provides endpoints for analyzing resumes against JDs, checking history,
health checks, and Supabase Authentication.
Run: uvicorn api:app --reload --port 8000
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from analyzer.scorer import ResumeScorer
from analyzer.models import GapAnalysisResult
from analyzer.database import ResumeDB

# Initialize FastAPI App
app = FastAPI(
    title="Resume Scorer & Gap Analysis API",
    description="BERT-powered semantic resume scoring, skill gap detection, ATS alignment, and Supabase Authentication.",
    version="1.1.0",
)

# Enable CORS for frontend integrations (React, Next.js, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Scorer Instance
scorer: Optional[ResumeScorer] = None


@app.on_event("startup")
def startup_event():
    """Pre-warm BERT model on application startup."""
    global scorer
    print("🚀 Initializing BERT Scorer Engine...")
    scorer = ResumeScorer()
    print("✅ BERT Model Ready!")


# ─── Pydantic Request Models ────────────────────────────────────

class TextAnalysisRequest(BaseModel):
    resume_text: str
    jd_text: str
    candidate_name: Optional[str] = None
    job_title: Optional[str] = None
    user_id: Optional[str] = None
    save_to_db: bool = True


class AuthRequest(BaseModel):
    email: str
    password: str


# ─── Health Endpoints ───────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "Resume Scorer & Gap Analysis API",
        "docs_url": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    db_status = "unconfigured"
    try:
        db = ResumeDB()
        db_status = "connected" if db.health_check() else "error"
    except Exception:
        db_status = "not_connected"

    return {
        "status": "healthy",
        "bert_model_loaded": scorer is not None and scorer.bert.is_loaded,
        "database": db_status,
    }


# ─── Auth Endpoints ─────────────────────────────────────────────

@app.post("/api/auth/register", tags=["Authentication"])
def register(request: AuthRequest):
    """Register a new user account with Supabase Auth."""
    try:
        db = ResumeDB()
        res = db.sign_up(request.email, request.password)
        if not res.get("success"):
            raise HTTPException(status_code=400, detail=res.get("error", "Registration failed"))
        return {
            "status": "success",
            "message": res.get("message"),
            "email": res.get("email"),
            "user_id": res.get("user_id"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/login", tags=["Authentication"])
def login(request: AuthRequest):
    """Authenticate an existing user with email and password."""
    try:
        db = ResumeDB()
        res = db.sign_in(request.email, request.password)
        if not res.get("success"):
            raise HTTPException(status_code=401, detail=res.get("error", "Invalid email or password"))
        session = res.get("session")
        return {
            "status": "success",
            "email": res.get("email"),
            "user_id": res.get("user_id"),
            "access_token": session.access_token if session else None,
            "token_type": "bearer",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/logout", tags=["Authentication"])
def logout():
    """Sign out the current session."""
    try:
        db = ResumeDB()
        db.sign_out()
        return {"status": "success", "message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Analysis Endpoints ─────────────────────────────────────────

@app.post("/api/analyze/text", response_model=GapAnalysisResult, tags=["Analysis"])
def analyze_text(request: TextAnalysisRequest):
    """
    Analyze raw resume text against raw job description text.
    """
    if not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
    if not request.jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    try:
        result = scorer.analyze(request.resume_text, request.jd_text)

        # Save to Supabase if enabled
        if request.save_to_db:
            try:
                db = ResumeDB()
                r_row = db.save_resume(
                    raw_text=request.resume_text,
                    candidate_name=request.candidate_name,
                    user_id=request.user_id,
                )
                j_row = db.save_job_description(
                    raw_text=request.jd_text,
                    title=request.job_title,
                    user_id=request.user_id,
                )
                if r_row.get("id") and j_row.get("id"):
                    db.save_analysis(
                        resume_id=r_row["id"],
                        jd_id=j_row["id"],
                        result_obj=result,
                        user_id=request.user_id,
                    )
            except Exception as e:
                print(f"Warning: Failed to save to Supabase: {e}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/file", response_model=GapAnalysisResult, tags=["Analysis"])
async def analyze_file(
    resume_file: UploadFile = File(..., description="Resume file (PDF or TXT)"),
    jd_text: str = Form(..., description="Job Description text"),
    candidate_name: Optional[str] = Form(None),
    job_title: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    save_to_db: bool = Form(True),
):
    """
    Upload a resume file (PDF/TXT) and compare it against a Job Description.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description text cannot be empty.")

    suffix = Path(resume_file.filename).suffix.lower()
    if suffix not in [".pdf", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(resume_file.file, tmp)
        tmp_path = tmp.name

    try:
        result = scorer.analyze(tmp_path, jd_text)

        # Save to Supabase if enabled
        if save_to_db:
            try:
                db = ResumeDB()
                raw_resume = scorer.resume_parser.extract_text(tmp_path)
                r_row = db.save_resume(
                    raw_text=raw_resume,
                    file_name=resume_file.filename,
                    candidate_name=candidate_name,
                    user_id=user_id,
                )
                j_row = db.save_job_description(
                    raw_text=jd_text,
                    title=job_title,
                    user_id=user_id,
                )
                if r_row.get("id") and j_row.get("id"):
                    db.save_analysis(
                        resume_id=r_row["id"],
                        jd_id=j_row["id"],
                        result_obj=result,
                        user_id=user_id,
                    )
            except Exception as e:
                print(f"Warning: Failed to save to Supabase: {e}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/history", tags=["History"])
def get_history(limit: int = 20, user_id: Optional[str] = None):
    """
    Fetch past resume evaluations from Supabase database. Optionally filtered by user_id.
    """
    try:
        db = ResumeDB()
        return db.list_analyses(limit=limit, user_id=user_id)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
