# BiasGuard — Improvement Plan

> **Reviewed:** April 25, 2026  
> **Reviewer:** Senior Software Architect Consultant  
> **Status:** Implementation Ready — All tasks are agent-assignable

---

## Executive Summary

BiasGuard is a well-structured LangGraph-powered RAG pipeline with solid foundations:
multi-stage Dockerfiles, Pydantic v2 models, structured logging, Prometheus metrics, and
meaningful test coverage for the scorer and rewrite utilities.

The following plan addresses **concrete, confirmed problems** found during code review — not
theoretical concerns. Each section lists the exact file and line(s) involved.

---

## 1. API — Error Handling Anti-Pattern

**Severity: HIGH** | File: `api/main.py`

### Problem
The `/analyze` endpoint catches all exceptions and returns `HTTP 200` with `success=False`:

```python
except Exception as e:
    return AnalyzeResponse(success=False, error=f"Analysis failed: {str(e)}")
```

This breaks standard HTTP semantics. Clients cannot distinguish success from failure via
status code, breaking monitoring dashboards, load balancers, and retry logic.

### Fix
- Raise `HTTPException(status_code=503, detail=...)` for LLM/pipeline failures.
- Raise `HTTPException(status_code=500, detail=...)` for unexpected errors.
- Return `200` **only** on actual success.

### Agent Task
> **TASK-1** — In `api/main.py`, replace the bare `except Exception` block in `analyze_document()`
> with proper `HTTPException` raises: 503 for orchestration failures, 500 for unexpected errors.
> Update `test_api.py` to assert status 503/500 for failure paths.

---

## 2. API — CORS Headers Too Permissive

**Severity: MEDIUM** | File: `api/main.py`

### Problem
```python
allow_headers=["*"],
```
Allowing all request headers is a CORS misconfiguration that widens the attack surface.

### Fix
Restrict to the minimum needed headers:
```python
allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-API-Key"],
```

### Agent Task
> **TASK-2** — In `api/main.py`, replace `allow_headers=["*"]` with an explicit list of
> required headers. Add the allowed list to `Settings` as `cors_allowed_headers` so it is
> configurable via env.

---

## 3. API — Missing Request Tracing Header

**Severity: MEDIUM** | File: `api/main.py`

### Problem
The logging middleware logs requests but never emits a `X-Request-ID` response header.
Without this, correlating logs to specific API calls in Grafana/Phoenix is manual work.

### Fix
Generate a UUID per request and attach it to the response header and structlog context.

### Agent Task
> **TASK-3** — In the `logging_middleware` in `api/main.py`, generate `request_id = uuid.uuid4()`
> and add `response.headers["X-Request-ID"] = str(request_id)`. Bind it to the structlog context.

---

## 4. Security — Default Secret Key Not Validated

**Severity: HIGH** | File: `config/settings.py`

### Problem
```python
api_secret_key: str = "change-me"
```
The default secret key is never validated. If deployed without setting the env var, the
API runs in production with a well-known secret.

### Fix
Add a `model_validator` that raises `ValueError` when `api_debug=False` and the key is
still the default `"change-me"`.

### Agent Task
> **TASK-4** — In `config/settings.py`, add a `@model_validator(mode="after")` that raises
> `ValueError("api_secret_key must be changed from the default in production")` when
> `not self.api_debug and self.api_secret_key == "change-me"`.

---

## 5. Docker Compose — Frontend Missing `depends_on`

**Severity: MEDIUM** | File: `docker-compose.yml`

### Problem
The `frontend` service has no `depends_on: api` dependency. On fresh `docker compose up`,
the Streamlit frontend can start before the API is ready, causing connection errors on
the welcome screen.

### Fix
Add `depends_on` and resource limits to both `frontend` and `api`:

```yaml
frontend:
  depends_on:
    api:
      condition: service_healthy
  deploy:
    resources:
      limits:
        memory: 512m
```

### Agent Task
> **TASK-5** — In `docker-compose.yml`, add `depends_on: api: condition: service_healthy`
> to the `frontend` service. Add `deploy.resources.limits` (memory: 1g for api, 512m for
> frontend, 512m for phoenix). Add a healthcheck to the `frontend` service using
> `curl -f http://localhost:8501/_stcore/health`.

---

## 6. Missing `.env.example`

**Severity: MEDIUM** | Missing file

### Problem
There is no `.env.example` in the repository. New contributors have no template to
start from, and required variables are not documented in one place.

### Fix
Create `.env.example` with all `Settings` fields documented, safe placeholder values,
and section comments that match the settings groups.

### Agent Task
> **TASK-6** — Create `.env.example` covering all fields from `config/settings.py`,
> organised by section (LLM, Embeddings, Vector DB, API, Observability, Logging, Features).
> Mark required fields with `# REQUIRED` and optional with their defaults.

---

## 7. No Abstract Base Class for Agents

**Severity: LOW** | Directory: `agents/`

### Problem
`RetrieverAgent`, `AnalyzerAgent`, `MitigatorAgent`, and `ScorerAgent` share no common
interface. Adding a new agent requires inspecting existing agents to discover conventions.
There is no contract enforced by the type system.

### Fix
Create `agents/base.py` with an `AbstractBiasAgent` ABC that defines the expected interface.

### Agent Task
> **TASK-7** — Create `agents/base.py` with an `AbstractBiasAgent` ABC using `abc.ABC`.
> Define abstract methods that each agent type should implement. Do **not** force all four
> existing agents to inherit immediately — make the base class available for new agents and
> optionally refactor one agent as a reference implementation.

---

## 8. Missing Unit Tests — Agents

**Severity: HIGH** | Directory: `tests/unit/`

### Problem
Only `ScorerAgent` and `rewrite_utils` have unit tests. `RetrieverAgent`, `AnalyzerAgent`,
and `MitigatorAgent` have zero test coverage. The three untested agents contain the most
complex logic in the system (LLM calls, vector search, JSON parsing).

### Gaps identified
| Agent | Test File | Status |
|-------|-----------|--------|
| `RetrieverAgent` | — | **MISSING** |
| `AnalyzerAgent`  | — | **MISSING** |
| `MitigatorAgent` | — | **MISSING** |
| `ScorerAgent`    | `test_scorer.py` | ✅ Good |
| `rewrite_utils`  | `test_orchestrator.py` | ✅ Good |

### Fix
Add mocked unit tests using `pytest-mock`. Mock `chromadb`, LLM calls, and `get_settings`.

### Agent Task
> **TASK-8** — Create `tests/unit/test_retriever_agent.py` with mocked `BiasDB.similarity_search`.
> Test: successful retrieval, empty result, chunking behaviour, deduplication.  
> Create `tests/unit/test_analyzer_agent.py` with mocked LLM. Test: valid JSON parse, malformed
> LLM response (graceful fallback), empty text, prompt formatting.  
> Create `tests/unit/test_mitigator_agent.py` with mocked LLM. Test: rewrites generated,
> all instances covered, malformed response fallback.

---

## 9. Missing Unit Tests — Config Validation

**Severity: MEDIUM** | `tests/unit/`

### Problem
No tests verify that settings are correctly loaded from env, that validators fire, or
that the production `api_secret_key` guard works.

### Agent Task
> **TASK-9** — Create `tests/unit/test_config.py` testing:
> - Default values load correctly.
> - `get_allowed_origins()` splits comma-separated origins.
> - `get_active_llm_api_key()` returns the right key per provider.
> - After TASK-4, the production `api_secret_key` validator raises on default value.

---

## 10. Missing Integration Tests — KB + Examples Endpoints

**Severity: MEDIUM** | `tests/integration/test_api.py`

### Problem
Integration tests cover `/health` and `/analyze` but not:
- `GET /kb/stats`
- `POST /kb/ingest`
- `GET /examples`
- `GET /metrics` (Prometheus)
- Error paths for `/analyze` (pipeline failure → 503)

### Agent Task
> **TASK-10** — Add a `TestKBEndpoints` class to `tests/integration/test_api.py` covering
> `/kb/stats` and `/kb/ingest`. Add `TestMetricsEndpoint` for `/metrics`. Extend
> `TestAnalyzeEndpoint` with a test for orchestrator failure returning HTTP 503 (after TASK-1).

---

## 11. CI/CD — No GitHub Actions Workflow

**Severity: MEDIUM** | Missing: `.github/workflows/`

### Problem
No automated CI runs on push or PR. This means linting errors, broken tests, and
coverage regressions are only caught locally.

### Fix
Create a GitHub Actions workflow that:
1. Runs `ruff` lint and format check
2. Runs `mypy` type checking
3. Runs `pytest --cov` with coverage reporting
4. Fails if coverage drops below the `fail_under = 70` threshold in `pyproject.toml`

### Agent Task
> **TASK-11** — Create `.github/workflows/ci.yml` with a `test` job that:
> - Runs on push to `main` and on all PRs.
> - Sets up Python 3.11, installs `requirements-dev.txt`.
> - Runs `ruff check .` and `ruff format --check .`.
> - Runs `mypy agents/ api/ config/ monitoring/`.
> - Runs `pytest --cov --cov-report=xml` and uploads coverage artifact.

---

## 12. `rewrite_utils` — Identical Span Edge Case

**Severity: LOW** | `agents/rewrite_utils.py`

### Problem
If two bias instances share the **exact same span text**, only the first is replaced
(correct behaviour via `replace(span, replacement, 1)`), but the second instance silently
produces no rewrite. There is no log warning or test covering this case.

### Agent Task
> **TASK-12** — Add a test in `tests/unit/test_orchestrator.py` for two instances with
> identical spans. Add a `logger.warning` in `rewrite_utils.py` when a span is no longer
> found in the document (indicating it was already consumed).

---

## 13. Docker — Frontend Dockerfile Missing Healthcheck

**Severity: LOW** | `docker/Dockerfile.frontend`

### Problem
`Dockerfile.api` has an inline `HEALTHCHECK` directive; `Dockerfile.frontend` does not.
Relying solely on Compose-level healthcheck means the image is not self-contained.

### Agent Task
> **TASK-13** — Add a `HEALTHCHECK` directive to `docker/Dockerfile.frontend` using
> `curl -f http://localhost:8501/_stcore/health || exit 1`.

---

## Priority Order for Agent Execution

| Priority | Task | Impact |
|----------|------|--------|
| 🔴 Critical | TASK-1 | Breaks client monitoring |
| 🔴 Critical | TASK-4 | Security vulnerability in production |
| 🟠 High | TASK-8 | 3 agents with zero test coverage |
| 🟠 High | TASK-11 | No automated quality gate |
| 🟡 Medium | TASK-2 | CORS security hardening |
| 🟡 Medium | TASK-3 | Observability improvement |
| 🟡 Medium | TASK-5 | Deployment reliability |
| 🟡 Medium | TASK-6 | Developer experience |
| 🟡 Medium | TASK-9 | Config test coverage |
| 🟡 Medium | TASK-10 | Integration test completeness |
| 🟢 Low | TASK-7 | Code architecture |
| 🟢 Low | TASK-12 | Edge case correctness |
| 🟢 Low | TASK-13 | Docker image completeness |

---

*All tasks above are scoped, actionable, and can be executed independently by an agent in any order respecting their dependencies (TASK-1 before TASK-10 error-path tests; TASK-4 before TASK-9 validation tests).*
