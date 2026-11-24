"""
Intelligent Skill and Keyword Extractor with Hierarchical Taxonomy.
Extracts hard skills, soft skills, tools, frameworks, and methodologies
using regular expressions, token boundary checks, and synonym normalization.
"""

import re
from typing import Dict, List, Set, Tuple


class SkillExtractor:
    """Extracts skills and technical keywords mapped to structured categories."""

    SKILL_TAXONOMY: Dict[str, List[str]] = {
        "programming_languages": [
            "python", "java", "c++", "c#", "c", "javascript", "typescript", "golang", "go",
            "rust", "kotlin", "swift", "scala", "r", "php", "ruby", "sql", "bash", "shell",
            "powershell", "matlab", "dart", "html", "css", "sass"
        ],
        "machine_learning_ai": [
            "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy", "scipy",
            "hugging face", "transformers", "bert", "roberta", "llm", "large language models",
            "rag", "langchain", "llamaindex", "opencv", "computer vision", "nlp",
            "natural language processing", "deep learning", "machine learning", "reinforcement learning",
            "xgboost", "lightgbm", "catboost", "spacy", "nltk", "mlflow", "dvc", "wandb",
            "chromadb", "pinecone", "faiss", "weaviate", "qdrant", "vector databases", "genai", "generative ai"
        ],
        "cloud_and_devops": [
            "aws", "amazon web services", "gcp", "google cloud", "azure", "docker",
            "kubernetes", "k8s", "ci/cd", "continuous integration", "jenkins", "github actions",
            "gitlab ci", "terraform", "ansible", "prometheus", "grafana", "linux", "helm",
            "nginx", "serverless", "cloudformation"
        ],
        "web_and_backend": [
            "fastapi", "flask", "django", "node.js", "nodejs", "express", "express.js",
            "spring boot", "asp.net", "rest api", "restful api", "rest", "graphql", "grpc",
            "microservices", "celery", "redis", "rabbitmq", "apache kafka", "kafka",
            "websockets", "react", "next.js", "vue", "angular", "tailwind css", "redux"
        ],
        "databases_and_data_engineering": [
            "postgresql", "postgres", "mysql", "sqlite", "mongodb", "cassandra", "dynamodb",
            "elasticsearch", "neo4j", "apache spark", "pyspark", "hadoop", "snowflake",
            "bigquery", "databricks", "apache airflow", "airflow", "dbt", "etl", "data pipelines"
        ],
        "tools_and_methodologies": [
            "git", "github", "gitlab", "bitbucket", "jira", "agile", "scrum", "kanban",
            "tdd", "test driven development", "unit testing", "pytest", "selenium",
            "postman", "swagger", "system design", "object oriented programming", "oop"
        ],
        "soft_skills": [
            "leadership", "communication", "team collaboration", "collaboration",
            "problem solving", "critical thinking", "project management", "mentoring",
            "adaptability", "time management", "stakeholder management", "cross-functional"
        ],
    }

    # Normalized canonical display names
    CANONICAL_NAMES: Dict[str, str] = {
        "python": "Python",
        "java": "Java",
        "c++": "C++",
        "c#": "C#",
        "c": "C",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "golang": "Go",
        "go": "Go",
        "rust": "Rust",
        "kotlin": "Kotlin",
        "swift": "Swift",
        "scala": "Scala",
        "r": "R",
        "php": "PHP",
        "ruby": "Ruby",
        "sql": "SQL",
        "bash": "Bash",
        "shell": "Shell Scripting",
        "powershell": "PowerShell",
        "matlab": "MATLAB",
        "dart": "Dart",
        "html": "HTML5",
        "css": "CSS3",
        "sass": "Sass",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "keras": "Keras",
        "scikit-learn": "Scikit-Learn",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "hugging face": "Hugging Face",
        "transformers": "Transformers",
        "bert": "BERT",
        "roberta": "RoBERTa",
        "llm": "LLMs",
        "large language models": "LLMs",
        "rag": "RAG (Retrieval-Augmented Generation)",
        "langchain": "LangChain",
        "llamaindex": "LlamaIndex",
        "opencv": "OpenCV",
        "computer vision": "Computer Vision",
        "nlp": "NLP",
        "natural language processing": "NLP",
        "deep learning": "Deep Learning",
        "machine learning": "Machine Learning",
        "reinforcement learning": "Reinforcement Learning",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "spacy": "spaCy",
        "nltk": "NLTK",
        "mlflow": "MLflow",
        "dvc": "DVC",
        "wandb": "Weights & Biases",
        "chromadb": "ChromaDB",
        "pinecone": "Pinecone",
        "faiss": "FAISS",
        "weaviate": "Weaviate",
        "qdrant": "Qdrant",
        "vector databases": "Vector Databases",
        "genai": "Generative AI",
        "generative ai": "Generative AI",
        "aws": "AWS",
        "amazon web services": "AWS",
        "gcp": "GCP",
        "google cloud": "Google Cloud Platform",
        "azure": "Azure",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "k8s": "Kubernetes",
        "ci/cd": "CI/CD",
        "continuous integration": "CI/CD",
        "jenkins": "Jenkins",
        "github actions": "GitHub Actions",
        "gitlab ci": "GitLab CI",
        "terraform": "Terraform",
        "ansible": "Ansible",
        "prometheus": "Prometheus",
        "grafana": "Grafana",
        "linux": "Linux",
        "helm": "Helm",
        "nginx": "Nginx",
        "serverless": "Serverless",
        "cloudformation": "CloudFormation",
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "node.js": "Node.js",
        "nodejs": "Node.js",
        "express": "Express.js",
        "express.js": "Express.js",
        "spring boot": "Spring Boot",
        "asp.net": "ASP.NET",
        "rest api": "RESTful APIs",
        "restful api": "RESTful APIs",
        "rest": "REST APIs",
        "graphql": "GraphQL",
        "grpc": "gRPC",
        "microservices": "Microservices",
        "celery": "Celery",
        "redis": "Redis",
        "rabbitmq": "RabbitMQ",
        "apache kafka": "Kafka",
        "kafka": "Kafka",
        "websockets": "WebSockets",
        "react": "React",
        "next.js": "Next.js",
        "vue": "Vue.js",
        "angular": "Angular",
        "tailwind css": "Tailwind CSS",
        "redux": "Redux",
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "mysql": "MySQL",
        "sqlite": "SQLite",
        "mongodb": "MongoDB",
        "cassandra": "Cassandra",
        "dynamodb": "DynamoDB",
        "elasticsearch": "Elasticsearch",
        "neo4j": "Neo4j",
        "apache spark": "Apache Spark",
        "pyspark": "PySpark",
        "hadoop": "Hadoop",
        "snowflake": "Snowflake",
        "bigquery": "BigQuery",
        "databricks": "Databricks",
        "apache airflow": "Apache Airflow",
        "airflow": "Apache Airflow",
        "dbt": "dbt",
        "etl": "ETL",
        "data pipelines": "Data Pipelines",
        "git": "Git",
        "github": "GitHub",
        "gitlab": "GitLab",
        "bitbucket": "Bitbucket",
        "jira": "Jira",
        "agile": "Agile",
        "scrum": "Scrum",
        "kanban": "Kanban",
        "tdd": "TDD",
        "test driven development": "TDD",
        "unit testing": "Unit Testing",
        "pytest": "PyTest",
        "selenium": "Selenium",
        "postman": "Postman",
        "swagger": "Swagger / OpenAPI",
        "system design": "System Design",
        "object oriented programming": "OOP",
        "oop": "OOP",
        "leadership": "Leadership",
        "communication": "Communication",
        "team collaboration": "Team Collaboration",
        "collaboration": "Collaboration",
        "problem solving": "Problem Solving",
        "critical thinking": "Critical Thinking",
        "project management": "Project Management",
        "mentoring": "Mentoring",
        "adaptability": "Adaptability",
        "time management": "Time Management",
        "stakeholder management": "Stakeholder Management",
        "cross-functional": "Cross-Functional Collaboration",
    }

    def __init__(self):
        # Precompile regex patterns for each skill keyword
        self.patterns = {}
        for category, skills in self.SKILL_TAXONOMY.items():
            for skill in skills:
                # Escape special characters like ++, #, /, .
                escaped = re.escape(skill)
                # Word boundary check
                if skill in ["c", "r"]:
                    # Strict boundary for single characters
                    pattern = re.compile(rf"(?<![a-zA-Z0-9_]){escaped}(?![a-zA-Z0-9_])", re.IGNORECASE)
                elif skill.endswith("+") or skill.endswith("#"):
                    pattern = re.compile(rf"\b{escaped}(?![a-zA-Z0-9])", re.IGNORECASE)
                else:
                    pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                self.patterns[skill] = (category, pattern)

    def extract_skills(self, text: str) -> Dict[str, Set[str]]:
        """
        Extracts skills from text, categorized by domain.
        Returns dict: {category_name: set_of_canonical_skill_names}.
        """
        results: Dict[str, Set[str]] = {cat: set() for cat in self.SKILL_TAXONOMY.keys()}

        for skill_kw, (category, pattern) in self.patterns.items():
            if pattern.search(text):
                canonical = self.CANONICAL_NAMES.get(skill_kw, skill_kw.title())
                results[category].add(canonical)

        return results

    def extract_flat_skills(self, text: str) -> Set[str]:
        """Returns a flat set of all detected canonical skill names."""
        cat_dict = self.extract_skills(text)
        all_skills = set()
        for skills in cat_dict.values():
            all_skills.update(skills)
        return all_skills

    def get_skill_category(self, skill_name: str) -> str:
        """Find category of a skill."""
        lower_name = skill_name.lower()
        for cat, skills in self.SKILL_TAXONOMY.items():
            if lower_name in skills or any(lower_name == s.lower() for s in skills):
                return cat
            # Check canonical mapping
            for s in skills:
                if self.CANONICAL_NAMES.get(s, "").lower() == lower_name:
                    return cat
        return "other"
