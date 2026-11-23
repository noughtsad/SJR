from app.ingestion.models import JobPosting
from app.ingestion.normalize import (
    content_hash,
    dedupe_key,
    extract_skills,
    html_to_text,
    infer_seniority,
)
from app.ingestion.sources.remotive import RemotiveSource, parse_salary


def job(**kw):
    base = dict(
        source="t", external_id="1", title="Senior Python Engineer", company="Acme",
        location="Berlin", description="We build data pipelines in Python and Postgres." * 2,
    )  # fmt: skip
    return JobPosting(**{**base, **kw})


def test_content_hash_ignores_non_embedded_fields():
    assert content_hash(job()) == content_hash(job(url="https://x", external_id="999"))
    assert content_hash(job()) != content_hash(job(description="Totally different text, " * 3))


def test_dedupe_key_normalises_case_and_punctuation():
    assert dedupe_key(job()) == dedupe_key(job(title="senior  python engineer!", company="ACME", source="other"))
    assert dedupe_key(job()) != dedupe_key(job(location="Paris"))


def test_seniority():
    assert infer_seniority("Sr. Backend Developer") == "senior"
    assert infer_seniority("Staff ML Engineer") == "lead"
    assert infer_seniority("Software Engineering Intern") == "intern"
    assert infer_seniority("Backend Developer") is None
    assert infer_seniority("Backend Developer", hint="Entry level") == "junior"


def test_skills_precision():
    s = extract_skills("Senior Java developer", "Spring Boot, Kafka, AWS. No JavaScript.")
    assert {"java", "spring", "kafka", "aws", "javascript"} <= set(s)
    assert "java" not in extract_skills("JavaScript only")
    assert "c++" in extract_skills("Modern C++ and Rust")
    assert "go" not in extract_skills("Ready to go the extra mile")


def test_html_to_text():
    assert html_to_text("<p>Hello&nbsp;<b>world</b></p><script>x()</script><ul><li>a</li></ul>").split() == [
        "Hello", "world", "a",
    ]  # fmt: skip


def test_parse_salary():
    assert parse_salary("$120k - $150k") == (120000, 150000, "USD")
    assert parse_salary("50,000 - 70,000 EUR") == (50000, 70000, "EUR")
    assert parse_salary("") == (None, None, None)
    assert parse_salary("competitive") == (None, None, None)


def test_remotive_mapping():
    raw = {
        "id": 42, "url": "https://remotive.com/x", "title": "Lead Data Engineer", "company_name": "Co",
        "category": "Data", "tags": ["python", "airflow"], "job_type": "full_time",
        "publication_date": "2026-11-01T10:00:00", "candidate_required_location": "Europe",
        "salary": "$100k-$130k", "description": "<p>Build <b>pipelines</b> with dbt and Snowflake.</p>",
    }  # fmt: skip
    p = RemotiveSource().to_posting(raw)
    assert p.external_id == "42" and p.is_remote and p.seniority == "lead"
    assert {"python", "airflow", "dbt", "snowflake"} <= set(p.skills)
    assert p.salary_min == 100000 and p.posted_at.tzinfo is not None
