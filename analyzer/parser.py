"""
Document and Text Parser for Resumes and Job Descriptions.
Supports PDF (via pypdf) and plain text with section segmentation and bullet extraction.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import pypdf


class ResumeParser:
    """Parses resumes in PDF or plain text format into structured sections."""

    SECTION_PATTERNS = {
        "summary": [
            r"\b(professional\s+summary|executive\s+summary|summary|profile|about\s+me|career\s+objective|objective)\b"
        ],
        "skills": [
            r"\b(technical\s+skills|core\s+competencies|skills\s+(&|and)\s+technologies|skills\s+(&|and)\s+tools|key\s+skills|skills|technologies|expertise)\b"
        ],
        "experience": [
            r"\b(work\s+experience|professional\s+experience|employment\s+history|experience|work\s+history|career\s+history)\b"
        ],
        "education": [
            r"\b(education|academic\s+background|educational\s+qualifications|academic\s+qualifications|degrees)\b"
        ],
        "projects": [
            r"\b(projects|key\s+projects|personal\s+projects|academic\s+projects|notable\s+projects)\b"
        ],
        "certifications": [
            r"\b(certifications|certificates|licenses|professional\s+certifications|credentials)\b"
        ],
    }

    def __init__(self):
        pass

    def extract_text(self, file_source: Union[str, Path, bytes]) -> str:
        """
        Extract text from file path (PDF or TXT) or raw string/bytes.
        """
        if isinstance(file_source, bytes):
            # Try reading as PDF bytes
            try:
                import io
                reader = pypdf.PdfReader(io.BytesIO(file_source))
                text_pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(text_pages).strip()
            except Exception:
                return file_source.decode("utf-8", errors="ignore").strip()

        path = Path(file_source)
        if path.is_file():
            if path.suffix.lower() == ".pdf":
                reader = pypdf.PdfReader(str(path))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages).strip()
            else:
                return path.read_text(encoding="utf-8", errors="ignore").strip()

        # If it's already a raw text string
        return str(file_source).strip()

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw text."""
        if not text:
            return ""
        # Replace non-standard bullet characters with standard dash
        text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\uF0B7\u25CF]", "\n- ", text)
        # Normalize carriage returns and tabs
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        # Remove consecutive empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_sections(self, text: str) -> Dict[str, str]:
        """
        Segment resume text into known sections.
        Returns a dict mapping section name -> section text.
        """
        cleaned = self.clean_text(text)
        lines = cleaned.split("\n")
        
        # Build regex for detecting any section header
        section_headers = {}
        for section, patterns in self.SECTION_PATTERNS.items():
            for pat in patterns:
                section_headers[re.compile(r"^\s*#*\s*" + pat + r"\s*[:\-]?\s*$", re.IGNORECASE)] = section
                section_headers[re.compile(r"^\s*#*\s*" + pat + r"\s*$", re.IGNORECASE)] = section

        current_section = "summary"
        sections: Dict[str, List[str]] = {
            "summary": [],
            "skills": [],
            "experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "other": [],
        }

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            matched_section = None
            # Check if this line looks like a header (short, matches section pattern)
            if len(line_stripped) < 50:
                for regex, sec_name in section_headers.items():
                    if regex.match(line_stripped):
                        matched_section = sec_name
                        break

            if matched_section:
                current_section = matched_section
            else:
                sections[current_section].append(line)

        result = {}
        for sec_name, sec_lines in sections.items():
            content = "\n".join(sec_lines).strip()
            if content:
                result[sec_name] = content

        return result

    def extract_bullets_and_sentences(self, text: str) -> List[str]:
        """
        Split text into discrete bullets and sentences for granular embedding analysis.
        """
        cleaned = self.clean_text(text)
        items = []

        for line in cleaned.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading bullet markers
            line = re.sub(r"^[-*•–—]\s*", "", line)
            # If line is longer than 15 chars, split into sentences or add line
            if len(line) > 15:
                # Split on sentence terminals if multi-sentence line
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
                for s in sentences:
                    s_clean = s.strip()
                    if len(s_clean) > 15:
                        items.append(s_clean)
            elif len(line) >= 3:
                items.append(line)

        return items


class JobDescriptionParser:
    """Parses Job Descriptions into requirements, qualifications, and core duties."""

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """Clean JD text."""
        if not text:
            return ""
        text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\uF0B7\u25CF]", "\n- ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_requirements(self, jd_text: str) -> List[str]:
        """
        Extract specific requirement bullet points or qualification lines from JD.
        """
        cleaned = self.clean_text(jd_text)
        lines = cleaned.split("\n")
        requirements = []

        is_in_req_section = False
        req_section_triggers = re.compile(
            r"\b(requirements|qualifications|what\s+you('ll|\s+will)\s+need|who\s+you\s+are|skills|responsibilities|must\s+have|nice\s+to\s+have|preferred\s+qualifications|basic\s+qualifications)\b",
            re.IGNORECASE
        )

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if this line triggers a requirements block
            if len(line_str) < 60 and req_section_triggers.search(line_str):
                is_in_req_section = True
                continue

            # Bullet points or lines within requirement section
            is_bullet = line_str.startswith(("-", "*", "•", "–")) or re.match(r"^\d+\.\s+", line_str)
            clean_item = re.sub(r"^[-*•–—\d.]+\s*", "", line_str).strip()

            if (is_in_req_section or is_bullet) and len(clean_item) > 15:
                # Filter out generic company intros or benefits lines if needed
                if not re.search(r"\b(equal\s+opportunity|benefits|401k|healthcare|pto|bonus)\b", clean_item, re.IGNORECASE):
                    requirements.append(clean_item)
            elif len(clean_item) > 25 and any(kw in clean_item.lower() for kw in ["experience", "proficient", "knowledge", "degree", "ability to", "familiarity", "hands-on", "track record", "proven"]):
                requirements.append(clean_item)

        # Fallback if no distinct requirements could be detected: split into sentences
        if len(requirements) < 3:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
            requirements = [s.strip() for s in sentences if len(s.strip()) > 20 and not re.search(r"\b(about us|company|perks|apply now)\b", s, re.IGNORECASE)]

        return requirements[:25]  # Cap to most relevant top 25 requirements to maintain high signal
