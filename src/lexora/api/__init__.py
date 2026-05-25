"""FastAPI audit UI + JSON API.

Routes:
    GET  /healthz
    GET  /jurisdictions
    GET  /jurisdictions/{iso}/indicators/{id}/citations
    GET  /citations/{id}
    GET  /audit/{citation_id}   → side-by-side PDF/HTML + extracted span
"""
