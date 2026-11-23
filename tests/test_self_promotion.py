import pytest

from app.ingestion.normalize import looks_like_self_promotion

# Paraphrased from patterns seen in HN "Who is hiring" threads (names are invented).
SELF_PROMO = [
    "I'm Alex Doe, a Full-Stack & AI Engineer at a YC-backed startup. I have hands-on experience shipping APIs.",
    "I am iOS developer with 10 years of experience working on streaming apps. I also build my own productivity app.",
    "I'm applying for the Growth Lead role, specifically the lifecycle automation piece. Milestone 1: instrument...",
    "Jane Roe | Software Engineer / Backend Engineer | India (REMOTE / ONSITE) I'm a software engineer focused on "
    "backend development with Java and Python. I enjoy building distributed systems.",
    "Hi! I'm currently looking for a job as a data analyst. My resume is linked below.",
]

REAL_JOBS = [
    "Acme | Senior Backend Engineer | REMOTE (US) | $150k-$190k + equity. We're hiring engineers to build our API.",
    "BigCo | Senior PM | Toronto - Hybrid. I am a Recruiter at BigCo, actively hiring for this role.",
    "Stealth AI | Full Stack Engineer | NYC | I'm an independent recruiter sourcing for a venture-backed startup.",
    "Startup | Remote | Hi! I am Max. I've built products for 13 years and I'm looking for founding team members.",
    "Factory | Security Engineer | Onsite | San Francisco. Come work with me! I'm the security lead here.",
    "Vohuman | San Francisco | Full time | Hey folks, i'm the founder. We build agents for enterprises.",
    "Our client is looking for a QA Engineer on a remote contract basis. I have 3 openings.",
    "We are a small team building developer tools. You will own our Rust backend. Apply at example.com/jobs.",
]


@pytest.mark.parametrize("text", SELF_PROMO)
def test_flags_self_promotion(text):
    assert looks_like_self_promotion(text)


@pytest.mark.parametrize("text", REAL_JOBS)
def test_keeps_real_jobs(text):
    assert not looks_like_self_promotion(text)
