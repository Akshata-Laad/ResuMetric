"""
Tests for parser module.
"""

from pathlib import Path
from analyzer.parser import ResumeParser, JobDescriptionParser


def test_resume_parser_clean_text():
    parser = ResumeParser()
    raw = "• Python\n\r\n• PyTorch\n\n\n• Docker"
    cleaned = parser.clean_text(raw)
    assert "Python" in cleaned
    assert "PyTorch" in cleaned
    assert "\r" not in cleaned


def test_resume_parser_sections():
    parser = ResumeParser()
    sample_text = """
    John Doe
    john@example.com

    TECHNICAL SKILLS
    Python, PyTorch, Docker, Kubernetes

    WORK EXPERIENCE
    Software Engineer at Tech Corp (2021 - Present)
    Built backend APIs using FastAPI.

    EDUCATION
    B.S. in Computer Science, MIT (2017 - 2021)
    """
    sections = parser.parse_sections(sample_text)
    assert "skills" in sections
    assert "experience" in sections
    assert "education" in sections
    assert "Python" in sections["skills"]
    assert "FastAPI" in sections["experience"]


def test_resume_parser_pdf():
    pdf_path = Path("sample_data/sample_resume.pdf")
    if pdf_path.exists():
        parser = ResumeParser()
        text = parser.extract_text(pdf_path)
        assert len(text) > 100
        assert "Alex Johnson" in text
        assert "Machine Learning" in text


def test_jd_parser_requirements():
    jd_parser = JobDescriptionParser()
    jd = """
    Software Engineer Job

    Requirements:
    - 3+ years experience with Python and Django
    - Hands-on experience with Docker and Kubernetes
    - Strong knowledge of PostgreSQL databases
    - Bachelor's degree in Computer Science
    """
    reqs = jd_parser.extract_requirements(jd)
    assert len(reqs) >= 3
    assert any("Python" in r for r in reqs)
