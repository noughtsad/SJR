"""Text cleaning, metadata extraction and hashing shared by all sources."""

import hashlib
import html
import re
import unicodedata

from bs4 import BeautifulSoup

from app.ingestion.models import JobPosting

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def html_to_text(raw: str | None) -> str:
    if not raw:
        return ""
    if "<" not in raw:
        return clean_text(html.unescape(raw))
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    return clean_text(text)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def _norm_key(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------


def embedding_text(job: JobPosting, max_chars: int = 4000) -> str:
    """The exact text that gets embedded. content_hash is computed over this,
    so any change that would alter the embedding triggers a re-embed, and
    changes that wouldn't (e.g. a new URL) don't."""
    parts = [job.title]
    if job.company:
        parts.append(f"Company: {job.company}")
    if job.location or job.is_remote:
        loc = job.location or ""
        parts.append(f"Location: {loc}{' (remote)' if job.is_remote else ''}".strip())
    if job.seniority:
        parts.append(f"Seniority: {job.seniority}")
    if job.skills:
        parts.append("Skills: " + ", ".join(job.skills))
    parts.append(job.description[:max_chars])
    return "\n".join(parts)


def content_hash(job: JobPosting) -> str:
    return hashlib.sha256(embedding_text(job).encode("utf-8")).hexdigest()


def dedupe_key(job: JobPosting) -> str:
    """Same role reposted by another source / under another id collapses to one key."""
    basis = "|".join(_norm_key(x) for x in (job.title, job.company, job.location))
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Metadata extraction (rule-based; cheap and explainable)
# ---------------------------------------------------------------------------

_SENIORITY_RULES: list[tuple[str, re.Pattern]] = [
    ("intern", re.compile(r"\b(intern|internship|trainee|apprentice)\b", re.I)),
    ("executive", re.compile(r"\b(chief|cto|ceo|cfo|vp|vice president|head of|director)\b", re.I)),
    ("lead", re.compile(r"\b(lead|staff|principal|architect|manager)\b", re.I)),
    ("senior", re.compile(r"\b(senior|sr\.?|iii|experienced)\b", re.I)),
    ("junior", re.compile(r"\b(junior|jr\.?|entry[- ]level|graduate|associate)\b", re.I)),
    ("mid", re.compile(r"\b(mid[- ]?level|intermediate|ii)\b", re.I)),
]

SENIORITY_LEVELS = ["intern", "junior", "mid", "senior", "lead", "executive"]


def infer_seniority(title: str, hint: str | None = None) -> str | None:
    """Title wins; a source's own label (e.g. LinkedIn's vague 'Mid-Senior level') is the fallback."""
    for level, pattern in _SENIORITY_RULES:
        if pattern.search(title):
            return level
    if hint:
        h = hint.lower()
        for level, words in (
            ("intern", ("intern",)),
            ("junior", ("entry", "associate", "junior")),
            ("mid", ("mid",)),
            ("senior", ("senior",)),
            ("executive", ("executive", "director")),
        ):
            if any(w in h for w in words):
                return level
    return None


_REMOTE = re.compile(r"\b(remote|work from home|wfh|anywhere|distributed)\b", re.I)


def looks_remote(*texts: str | None) -> bool:
    return any(t and _REMOTE.search(t) for t in texts)


# canonical skill -> regex alternatives. Kept deliberately small and
# high-precision; the LLM profile extractor (Layer 3) handles the long tail.
_SKILLS: dict[str, str] = {
    "python": r"python",
    "java": r"java(?!script)",
    "javascript": r"javascript|\bjs\b",
    "typescript": r"typescript|\bts\b",
    "go": r"golang|\bgo\b(?= developer| engineer|,|/| and)",
    "rust": r"rust",
    "c++": r"c\+\+|cpp",
    "c#": r"c#|\.net|dotnet",
    "ruby": r"ruby|rails",
    "php": r"php|laravel",
    "kotlin": r"kotlin",
    "swift": r"swift(?:ui)?",
    "scala": r"scala",
    "sql": r"sql",
    "postgresql": r"postgres(?:ql)?",
    "mysql": r"mysql",
    "mongodb": r"mongo(?:db)?",
    "redis": r"redis",
    "elasticsearch": r"elastic ?search|opensearch",
    "kafka": r"kafka",
    "spark": r"(?:py)?spark",
    "airflow": r"airflow",
    "dbt": r"\bdbt\b",
    "snowflake": r"snowflake",
    "react": r"react(?:\.js|js)?(?! native)",
    "react native": r"react native",
    "next.js": r"next\.?js",
    "vue": r"vue(?:\.js)?",
    "angular": r"angular",
    "node.js": r"node(?:\.js|js)?",
    "django": r"django",
    "flask": r"flask",
    "fastapi": r"fastapi",
    "spring": r"spring(?: boot)?",
    "graphql": r"graphql",
    "aws": r"\baws\b|amazon web services",
    "gcp": r"\bgcp\b|google cloud",
    "azure": r"azure",
    "docker": r"docker",
    "kubernetes": r"kubernetes|\bk8s\b",
    "terraform": r"terraform",
    "ci/cd": r"ci/cd|continuous integration",
    "linux": r"linux",
    "machine learning": r"machine learning|\bml\b",
    "deep learning": r"deep learning",
    "nlp": r"\bnlp\b|natural language processing",
    "computer vision": r"computer vision",
    "llm": r"\bllms?\b|large language models?",
    "rag": r"\brag\b|retrieval[- ]augmented",
    "pytorch": r"pytorch",
    "tensorflow": r"tensorflow",
    "scikit-learn": r"scikit[- ]learn|sklearn",
    "pandas": r"pandas",
    "langchain": r"langchain|langgraph",
    "data analysis": r"data analysis|data analytics",
    "tableau": r"tableau",
    "power bi": r"power ?bi",
    "excel": r"excel",
    "figma": r"figma",
    "seo": r"\bseo\b",
    "salesforce": r"salesforce",
    "product management": r"product management",
    "agile": r"agile|scrum",
}
_SKILL_PATTERNS = {k: re.compile(rf"(?<![\w+#.])(?:{v})(?![\w+#])", re.I) for k, v in _SKILLS.items()}


def extract_skills(*texts: str | None, extra: list[str] | None = None) -> list[str]:
    blob = "\n".join(t for t in texts if t)
    found = {k for k, p in _SKILL_PATTERNS.items() if p.search(blob)}
    for tag in extra or []:
        t = tag.strip().lower()
        if t in _SKILLS:
            found.add(t)
    return sorted(found)
