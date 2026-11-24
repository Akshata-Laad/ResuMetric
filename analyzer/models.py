"""
Data models and schemas for Resume Scorer & Gap Analysis.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class SkillMatch(BaseModel):
    name: str
    category: str
    status: str  # "matched", "missing", "extra"
    context: Optional[str] = None


class RequirementMatch(BaseModel):
    requirement: str
    similarity_score: float  # 0.0 - 100.0
    status: str  # "strong_match", "moderate_match", "gap"
    best_matching_text: Optional[str] = None
    section_source: Optional[str] = None


class SectionAnalysis(BaseModel):
    name: str
    present: bool
    word_count: int
    content_preview: str
    similarity_to_jd: float  # 0.0 - 100.0


class ExperienceAnalysis(BaseModel):
    required_years: Optional[float] = None
    candidate_years: Optional[float] = None
    status: str = "undetermined"  # "meets", "exceeds", "underqualified", "undetermined"
    details: str = ""


class EducationAnalysis(BaseModel):
    required_level: Optional[str] = None
    candidate_level: Optional[str] = None
    status: str = "undetermined"  # "meets", "exceeds", "underqualified", "undetermined"
    details: str = ""


class CategorySkillStats(BaseModel):
    total_jd: int = 0
    matched: int = 0
    missing: int = 0
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)


class GapAnalysisResult(BaseModel):
    # Overall and sub-scores (0 - 100)
    overall_score: float
    semantic_score: float
    skills_score: float
    experience_score: float
    education_score: float
    
    # Rating label (e.g. Excellent Match, Good Match, Moderate Match, Low Match)
    match_tier: str
    summary: str

    # Skills Analysis
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    extra_skills: List[str] = Field(default_factory=list)
    skills_by_category: Dict[str, CategorySkillStats] = Field(default_factory=dict)
    
    # Granular Requirements Semantic Match
    requirement_matches: List[RequirementMatch] = Field(default_factory=list)
    strong_matches_count: int = 0
    moderate_matches_count: int = 0
    gaps_count: int = 0
    
    # Sections breakdown
    sections: List[SectionAnalysis] = Field(default_factory=list)

    # Experience & Education
    experience: ExperienceAnalysis = Field(default_factory=ExperienceAnalysis)
    education: EducationAnalysis = Field(default_factory=EducationAnalysis)

    # Actionable Recommendations
    recommendations: List[str] = Field(default_factory=list)
    bullet_improvement_suggestions: List[Dict[str, str]] = Field(default_factory=list)
    ats_formatting_tips: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
