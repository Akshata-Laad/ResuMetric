"""
Tests for skill extraction and taxonomy.
"""

from analyzer.skill_extractor import SkillExtractor


def test_skill_extraction_taxonomy():
    extractor = SkillExtractor()
    sample = "Proficient in Python, PyTorch, Docker, Kubernetes, AWS, PostgreSQL, and Git."
    skills_by_cat = extractor.extract_skills(sample)

    assert "Python" in skills_by_cat["programming_languages"]
    assert "PyTorch" in skills_by_cat["machine_learning_ai"]
    assert "Docker" in skills_by_cat["cloud_and_devops"]
    assert "Kubernetes" in skills_by_cat["cloud_and_devops"]
    assert "AWS" in skills_by_cat["cloud_and_devops"]
    assert "PostgreSQL" in skills_by_cat["databases_and_data_engineering"]
    assert "Git" in skills_by_cat["tools_and_methodologies"]


def test_skill_synonyms_and_edge_cases():
    extractor = SkillExtractor()
    sample = "Extensive experience using k8s, postgres, and scikit-learn alongside BERT."
    flat_skills = extractor.extract_flat_skills(sample)

    assert "Kubernetes" in flat_skills
    assert "PostgreSQL" in flat_skills
    assert "Scikit-Learn" in flat_skills
    assert "BERT" in flat_skills


def test_isolated_word_boundary():
    extractor = SkillExtractor()
    # 'c' or 'r' shouldn't match arbitrary words
    sample = "Car driver with cat pet and great reading ability."
    flat_skills = extractor.extract_flat_skills(sample)
    assert "C" not in flat_skills
    assert "R" not in flat_skills
