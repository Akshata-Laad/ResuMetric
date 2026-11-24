"""
Composite Scoring Engine & Recommendation Generator.
Computes weighted ATS score, sub-dimension scores, actionable recommendations,
and bullet point enhancement tips.
"""

from typing import Dict, List, Optional
from .models import (
    GapAnalysisResult,
    RequirementMatch,
    ExperienceAnalysis,
    EducationAnalysis,
    CategorySkillStats,
    SectionAnalysis,
)
from .parser import ResumeParser, JobDescriptionParser
from .bert_engine import BertEngine
from .skill_extractor import SkillExtractor
from .gap_analyzer import GapAnalyzer


class ResumeScorer:
    """Orchestrates parsing, embedding, gap detection, and composite scoring."""

    def __init__(
        self,
        bert_engine: Optional[BertEngine] = None,
        skill_extractor: Optional[SkillExtractor] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.bert = bert_engine or BertEngine()
        self.skill_extractor = skill_extractor or SkillExtractor()
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.gap_analyzer = GapAnalyzer(self.bert, self.skill_extractor)

        # Default ATS scoring weights (sum = 1.0)
        self.weights = weights or {
            "semantic": 0.35,
            "skills": 0.35,
            "experience": 0.15,
            "education": 0.15,
        }

    def analyze(self, resume_source: str, jd_source: str) -> GapAnalysisResult:
        """
        End-to-end analysis of resume vs job description.
        Accepts file paths, raw text, or byte streams.
        """
        # 1. Parse raw text
        resume_raw = self.resume_parser.extract_text(resume_source)
        jd_raw = self.jd_parser.clean_text(self.resume_parser.extract_text(jd_source))

        if not resume_raw.strip():
            raise ValueError("Resume content is empty. Please provide valid text or PDF.")
        if not jd_raw.strip():
            raise ValueError("Job description is empty. Please provide valid text.")

        # 2. Segment Resume and JD
        resume_sections = self.resume_parser.parse_sections(resume_raw)
        resume_bullets = self.resume_parser.extract_bullets_and_sentences(resume_raw)
        jd_requirements = self.jd_parser.extract_requirements(jd_raw)

        # 3. Semantic Similarity (Global BERT)
        global_semantic_sim = self.bert.compute_similarity(resume_raw, jd_raw) * 100.0

        # 4. Granular Requirements Gap Analysis
        req_matches = self.gap_analyzer.analyze_requirements(
            jd_requirements, resume_bullets, resume_sections
        )

        strong_count = sum(1 for m in req_matches if m.status == "strong_match")
        moderate_count = sum(1 for m in req_matches if m.status == "moderate_match")
        gap_count = sum(1 for m in req_matches if m.status == "gap")

        req_semantic_avg = (
            sum(m.similarity_score for m in req_matches) / len(req_matches)
            if req_matches
            else global_semantic_sim
        )
        # Combined semantic score
        semantic_score = round(0.5 * global_semantic_sim + 0.5 * req_semantic_avg, 1)

        # 5. Skills Gap Analysis
        matched_skills, missing_skills, extra_skills, cat_stats = (
            self.gap_analyzer.analyze_skills(resume_raw, jd_raw)
        )

        total_jd_skills = len(matched_skills) + len(missing_skills)
        if total_jd_skills > 0:
            skills_score = round((len(matched_skills) / total_jd_skills) * 100.0, 1)
        else:
            # If JD didn't explicitly mention standard taxonomy skills
            skills_score = round(semantic_score, 1)

        # 6. Experience Gap Analysis
        exp_analysis = self.gap_analyzer.analyze_experience(resume_raw, jd_raw)
        if exp_analysis.status in ["meets", "exceeds"]:
            exp_score = 95.0 if exp_analysis.status == "meets" else 100.0
        elif exp_analysis.status == "underqualified":
            if exp_analysis.required_years and exp_analysis.candidate_years:
                ratio = exp_analysis.candidate_years / max(1.0, exp_analysis.required_years)
                exp_score = round(min(85.0, max(30.0, ratio * 75.0)), 1)
            else:
                exp_score = 55.0
        else:
            exp_score = 75.0  # neutral

        # 7. Education Gap Analysis
        edu_analysis = self.gap_analyzer.analyze_education(resume_raw, jd_raw)
        if edu_analysis.status in ["meets", "exceeds"]:
            edu_score = 100.0
        elif edu_analysis.status == "underqualified":
            edu_score = 60.0
        else:
            edu_score = 80.0  # neutral

        # 8. Overall Weighted Composite Score
        overall_score = round(
            (self.weights["semantic"] * semantic_score)
            + (self.weights["skills"] * skills_score)
            + (self.weights["experience"] * exp_score)
            + (self.weights["education"] * edu_score),
            1,
        )

        # 9. Match Tier
        if overall_score >= 82.0:
            match_tier = "Strong Match (High ATS Alignment)"
        elif overall_score >= 68.0:
            match_tier = "Good Match (Competitive with Minor Gaps)"
        elif overall_score >= 50.0:
            match_tier = "Moderate Match (Notable Skill & Requirement Gaps)"
        else:
            match_tier = "Low Match (Significant Gaps Detected)"

        # 10. Summary Statement
        summary = (
            f"Resume achieved an overall score of {overall_score}% ({match_tier}). "
            f"Candidate matches {len(matched_skills)} of {total_jd_skills} required skills "
            f"and aligns with {strong_count} of {len(req_matches)} primary job requirements."
        )

        # 11. Section Relevance
        section_analyses = self.gap_analyzer.analyze_sections(resume_sections, jd_raw)

        # 12. Recommendations & Bullet Suggestions
        recommendations = self._generate_recommendations(
            missing_skills=missing_skills,
            gaps_count=gap_count,
            req_matches=req_matches,
            exp_analysis=exp_analysis,
            edu_analysis=edu_analysis,
            resume_sections=resume_sections,
        )

        bullet_suggestions = self._generate_bullet_suggestions(req_matches, resume_bullets)

        ats_tips = [
            "Use clear, standard section headings: 'Work Experience', 'Skills', 'Education', 'Projects'.",
            "Incorporate exact keywords from the job description naturally into project descriptions.",
            "Quantify achievements where possible (e.g., 'Improved model inference speed by 35%').",
            "Avoid multi-column tables, text boxes, or non-standard icons that can confuse ATS parsers.",
            "Ensure dates are formatted consistently (e.g., 'Jan 2023 - Present' or '2023 - 2024').",
        ]

        return GapAnalysisResult(
            overall_score=overall_score,
            semantic_score=semantic_score,
            skills_score=skills_score,
            experience_score=exp_score,
            education_score=edu_score,
            match_tier=match_tier,
            summary=summary,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            extra_skills=extra_skills,
            skills_by_category=cat_stats,
            requirement_matches=req_matches,
            strong_matches_count=strong_count,
            moderate_matches_count=moderate_count,
            gaps_count=gap_count,
            sections=section_analyses,
            experience=exp_analysis,
            education=edu_analysis,
            recommendations=recommendations,
            bullet_improvement_suggestions=bullet_suggestions,
            ats_formatting_tips=ats_tips,
            metadata={
                "resume_word_count": len(resume_raw.split()),
                "jd_word_count": len(jd_raw.split()),
                "total_requirements_analyzed": len(req_matches),
            },
        )

    def _generate_recommendations(
        self,
        missing_skills: List[str],
        gaps_count: int,
        req_matches: List[RequirementMatch],
        exp_analysis: ExperienceAnalysis,
        edu_analysis: EducationAnalysis,
        resume_sections: Dict[str, str],
    ) -> List[str]:
        recs = []

        # High priority: Missing Skills
        if missing_skills:
            top_missing = missing_skills[:6]
            recs.append(
                f"🚨 **Critical Missing Skills**: Add experience or projects involving {', '.join(top_missing)} to boost ATS keyword match."
            )

        # Requirement Gaps
        if gaps_count > 0:
            top_gap_reqs = [m.requirement for m in req_matches if m.status == "gap"][:2]
            for req in top_gap_reqs:
                recs.append(
                    f"🎯 **Address Requirement Gap**: The JD requires '{req[:90]}...' — demonstrate related practical experience in your bullet points."
                )

        # Experience gap
        if exp_analysis.status == "underqualified":
            recs.append(
                f"⏱️ **Experience Alignment**: {exp_analysis.details} Emphasize senior-level scope, complexity, and leadership to offset total years."
            )

        # Missing sections
        for sec in ["skills", "experience", "projects", "education"]:
            if not resume_sections.get(sec):
                recs.append(f"📄 **Section Missing**: Add an explicit '{sec.title()}' section to help ATS parsers classify your credentials.")

        # General enhancement
        recs.append("💡 **Action Verbs**: Begin bullet points with strong verbs (e.g., 'Architected', 'Spearheaded', 'Optimized', 'Deployed').")

        return recs

    def _generate_bullet_suggestions(
        self, req_matches: List[RequirementMatch], resume_bullets: List[str]
    ) -> List[Dict[str, str]]:
        suggestions = []
        # Find requirements with moderate match that can be strengthened
        moderate_matches = [m for m in req_matches if m.status == "moderate_match"][:3]

        for match in moderate_matches:
            if match.best_matching_text:
                suggestions.append({
                    "original": match.best_matching_text,
                    "target_requirement": match.requirement,
                    "tip": f"Enhance this bullet by directly mentioning technologies from: '{match.requirement[:70]}...'",
                })

        return suggestions
