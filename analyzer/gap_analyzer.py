"""
Semantic Gap Analyzer.
Performs sentence-level BERT requirement alignment, skill overlap calculation,
and experience / education gap detection.
"""

import re
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .models import (
    RequirementMatch,
    SkillMatch,
    SectionAnalysis,
    ExperienceAnalysis,
    EducationAnalysis,
    CategorySkillStats,
)
from .bert_engine import BertEngine
from .skill_extractor import SkillExtractor
from .parser import ResumeParser, JobDescriptionParser


class GapAnalyzer:
    """Analyzes semantic gaps between a resume and a job description."""

    STRONG_MATCH_THRESHOLD = 0.65
    MODERATE_MATCH_THRESHOLD = 0.45

    def __init__(self, bert_engine: Optional[BertEngine] = None, skill_extractor: Optional[SkillExtractor] = None):
        self.bert = bert_engine or BertEngine()
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()

    def analyze_requirements(
        self,
        jd_requirements: List[str],
        resume_bullets: List[str],
        resume_sections: Dict[str, str],
    ) -> List[RequirementMatch]:
        """
        Computes granular BERT semantic match for each JD requirement against resume points.
        """
        if not jd_requirements:
            return []

        # If resume has no bullets, use whole sections or sentences
        candidate_pool = resume_bullets if resume_bullets else [v for v in resume_sections.values() if v]
        if not candidate_pool:
            candidate_pool = ["Candidate has provided no details."]

        # Compute cosine similarity matrix: (num_reqs, num_bullets)
        sim_matrix = self.bert.compute_pairwise_matrix(jd_requirements, candidate_pool)

        matches: List[RequirementMatch] = []
        for i, req in enumerate(jd_requirements):
            if sim_matrix.size > 0:
                row_sims = sim_matrix[i]
                best_idx = int(np.argmax(row_sims))
                best_score = float(row_sims[best_idx])
                best_text = candidate_pool[best_idx]
            else:
                best_score = 0.0
                best_text = "No direct match found in resume."

            # Classify status
            if best_score >= self.STRONG_MATCH_THRESHOLD:
                status = "strong_match"
            elif best_score >= self.MODERATE_MATCH_THRESHOLD:
                status = "moderate_match"
            else:
                status = "gap"

            # Identify section origin if possible
            matched_sec = None
            for sec_name, sec_text in resume_sections.items():
                if best_text in sec_text:
                    matched_sec = sec_name
                    break

            matches.append(
                RequirementMatch(
                    requirement=req,
                    similarity_score=round(best_score * 100.0, 1),
                    status=status,
                    best_matching_text=best_text,
                    section_source=matched_sec,
                )
            )

        return matches

    def analyze_skills(
        self, resume_text: str, jd_text: str
    ) -> Tuple[List[str], List[str], List[str], Dict[str, CategorySkillStats]]:
        """
        Identifies matched, missing, and extra skills along with category breakdowns.
        """
        resume_skills_by_cat = self.skill_extractor.extract_skills(resume_text)
        jd_skills_by_cat = self.skill_extractor.extract_skills(jd_text)

        all_resume_skills = set()
        all_jd_skills = set()
        for s in resume_skills_by_cat.values():
            all_resume_skills.update(s)
        for s in jd_skills_by_cat.values():
            all_jd_skills.update(s)

        matched_skills = sorted(list(all_jd_skills.intersection(all_resume_skills)))
        missing_skills = sorted(list(all_jd_skills.difference(all_resume_skills)))
        extra_skills = sorted(list(all_resume_skills.difference(all_jd_skills)))

        # Category breakdown
        category_stats: Dict[str, CategorySkillStats] = {}
        all_categories = set(list(resume_skills_by_cat.keys()) + list(jd_skills_by_cat.keys()))

        for cat in all_categories:
            cat_jd = jd_skills_by_cat.get(cat, set())
            cat_res = resume_skills_by_cat.get(cat, set())

            c_matched = sorted(list(cat_jd.intersection(cat_res)))
            c_missing = sorted(list(cat_jd.difference(cat_res)))

            if cat_jd or cat_res:
                category_stats[cat] = CategorySkillStats(
                    total_jd=len(cat_jd),
                    matched=len(c_matched),
                    missing=len(c_missing),
                    matched_skills=c_matched,
                    missing_skills=c_missing,
                )

        return matched_skills, missing_skills, extra_skills, category_stats

    def analyze_experience(self, resume_text: str, jd_text: str) -> ExperienceAnalysis:
        """
        Detects required vs candidate years of experience.
        """
        # Find required years in JD
        req_pattern = re.compile(r"(\d{1,2})\+?\s*(?:to\s*(\d{1,2})\s*)?(?:years?|yrs?)(?:\s+of)?(?:\s+relevant)?\s+experience", re.IGNORECASE)
        jd_matches = req_pattern.findall(jd_text)

        required_years = None
        if jd_matches:
            # Take the minimum required years detected
            years_found = [float(m[0]) for m in jd_matches if m[0]]
            if years_found:
                required_years = max(years_found)  # e.g. "5+ years"

        # Find candidate years in resume
        candidate_years = None
        res_matches = req_pattern.findall(resume_text)
        if res_matches:
            years_found = [float(m[0]) for m in res_matches if m[0]]
            if years_found:
                candidate_years = max(years_found)

        # Date range heuristics if no explicit "X years" statement
        if candidate_years is None:
            year_matches = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", resume_text)]
            valid_years = [y for y in year_matches if 1990 <= y <= 2026]
            if len(valid_years) >= 2:
                span = max(valid_years) - min(valid_years)
                if 0 < span < 35:
                    candidate_years = float(span)

        status = "undetermined"
        details = "No explicit experience requirements identified."

        if required_years is not None and candidate_years is not None:
            if candidate_years >= required_years:
                status = "meets" if candidate_years == required_years else "exceeds"
                details = f"Candidate has ~{int(candidate_years)} years of experience, meeting/exceeding requirement of {int(required_years)} years."
            else:
                status = "underqualified"
                details = f"JD calls for ~{int(required_years)} years of experience; resume indicates ~{int(candidate_years)} years."
        elif required_years is not None:
            details = f"JD mentions {int(required_years)}+ years required; ensure your work dates clearly display this."
        elif candidate_years is not None:
            details = f"Resume highlights ~{int(candidate_years)} years of experience."

        return ExperienceAnalysis(
            required_years=required_years,
            candidate_years=candidate_years,
            status=status,
            details=details,
        )

    def analyze_education(self, resume_text: str, jd_text: str) -> EducationAnalysis:
        """
        Detects degree requirements and candidate qualifications.
        """
        levels = [
            ("phd", ["ph.d", "phd", "doctorate"]),
            ("master", ["master", "m.s.", "ms ", "m.tech", "mba", "msc"]),
            ("bachelor", ["bachelor", "b.s.", "bs ", "b.tech", "b.e.", "bsc", "undergraduate"]),
        ]

        def get_highest_degree(text: str) -> Optional[str]:
            lower = text.lower()
            for level_name, keywords in levels:
                for kw in keywords:
                    if re.search(rf"\b{re.escape(kw)}\b", lower):
                        return level_name
            return None

        req_degree = get_highest_degree(jd_text)
        cand_degree = get_highest_degree(resume_text)

        status = "undetermined"
        degree_rank = {"bachelor": 1, "master": 2, "phd": 3}

        if req_degree and cand_degree:
            cand_rank = degree_rank.get(cand_degree, 0)
            req_rank = degree_rank.get(req_degree, 0)
            if cand_rank >= req_rank:
                status = "meets" if cand_rank == req_rank else "exceeds"
                details = f"Education aligns: Candidate holds {cand_degree.title()} (Required: {req_degree.title()})."
            else:
                status = "underqualified"
                details = f"JD specifies {req_degree.title()} degree, while resume shows {cand_degree.title()}."
        elif req_degree:
            details = f"JD mentions {req_degree.title()} level qualification."
        elif cand_degree:
            details = f"Candidate highlights {cand_degree.title()} degree."
        else:
            details = "No strict degree requirement specified in JD."

        return EducationAnalysis(
            required_level=req_degree.title() if req_degree else None,
            candidate_level=cand_degree.title() if cand_degree else None,
            status=status,
            details=details,
        )

    def analyze_sections(self, resume_sections: Dict[str, str], jd_text: str) -> List[SectionAnalysis]:
        """
        Evaluates relevance of individual resume sections against the JD.
        """
        analyses = []
        standard_sections = ["summary", "skills", "experience", "projects", "education"]

        for sec in standard_sections:
            content = resume_sections.get(sec, "")
            present = bool(content.strip())
            word_count = len(content.split()) if present else 0
            preview = (content[:150] + "...") if len(content) > 150 else content

            # Compute section-level semantic similarity to JD
            similarity = 0.0
            if present:
                similarity = round(self.bert.compute_similarity(content, jd_text) * 100.0, 1)

            analyses.append(
                SectionAnalysis(
                    name=sec.title(),
                    present=present,
                    word_count=word_count,
                    content_preview=preview,
                    similarity_to_jd=similarity,
                )
            )

        return analyses
