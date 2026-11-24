"""
Tests for Supabase Auth and database operations interface.
"""

from analyzer.database import ResumeDB


def test_resume_db_interface():
    """Verify that ResumeDB exposes the expected auth methods."""
    assert hasattr(ResumeDB, "sign_up")
    assert hasattr(ResumeDB, "sign_in")
    assert hasattr(ResumeDB, "sign_out")
    assert hasattr(ResumeDB, "get_current_user")
    assert hasattr(ResumeDB, "save_resume")
    assert hasattr(ResumeDB, "save_job_description")
    assert hasattr(ResumeDB, "save_analysis")
    assert hasattr(ResumeDB, "list_analyses")
