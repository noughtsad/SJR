"""Metadata filters -> a parameterised SQL fragment over alias `j` (jobs)."""

from psycopg import sql

from app.search.models import SearchFilters


def build_filters(f: SearchFilters) -> tuple[sql.Composable, dict]:
    clauses: list[sql.Composable] = [sql.SQL("j.is_active")]
    params: dict = {}
    if f.remote is not None:
        clauses.append(sql.SQL("j.is_remote = %(f_remote)s"))
        params["f_remote"] = f.remote
    if f.location:
        clauses.append(sql.SQL("j.location ILIKE %(f_location)s"))
        params["f_location"] = f"%{f.location}%"
    if f.seniority:
        clauses.append(sql.SQL("j.seniority = ANY(%(f_seniority)s)"))
        params["f_seniority"] = [s.lower() for s in f.seniority]
    if f.skills:
        clauses.append(sql.SQL("j.skills && %(f_skills)s"))
        params["f_skills"] = [s.lower() for s in f.skills]
    if f.posted_within_days:
        clauses.append(sql.SQL("j.posted_at >= now() - make_interval(days => %(f_days)s)"))
        params["f_days"] = f.posted_within_days
    if f.sources:
        clauses.append(sql.SQL("j.source = ANY(%(f_sources)s)"))
        params["f_sources"] = f.sources
    return sql.SQL(" AND ").join(clauses), params
