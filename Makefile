# ============================================================
# BiasGuard Makefile
# ============================================================

.PHONY: help install dev-install ingest api frontend docker-up docker-down test lint format clean

# Default target
help:
	@echo "BiasGuard — Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install production dependencies"
	@echo "    make dev-install    Install dev + test dependencies"
	@echo "    make ingest         Ingest bias knowledge base into ChromaDB"
	@echo ""
	@echo "  Run:"
	@echo "    make api            Start FastAPI server (dev mode)"
	@echo "    make frontend       Start Streamlit frontend"
	@echo "    make docker-up      Start full stack with Docker Compose"
	@echo "    make docker-down    Stop Docker Compose stack"
	@echo ""
	@echo "  Quality:"
	@echo "    make test           Run all tests with coverage"
	@echo "    make test-unit      Run unit tests only"
	@echo "    make lint           Run ruff linter"
	@echo "    make format         Auto-format with ruff"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean          Remove build artifacts and __pycache__"
	@echo "    make analyze-demo   Run CLI analysis on sample JD"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt
	pre-commit install

ingest:
	@echo "Ingesting bias knowledge base..."
	curl -X POST "http://localhost:8000/kb/ingest?force=true"
	@echo ""
	@echo "✅ Ingestion complete"

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/streamlit_app.py --server.port 8501

docker-up:
	docker compose up --build -d
	@echo ""
	@echo "✅ BiasGuard stack started:"
	@echo "   API:       http://localhost:8000/docs"
	@echo "   Frontend:  http://localhost:8501"
	@echo "   Grafana:   http://localhost:3000 (admin/biasguard)"
	@echo "   Phoenix:   http://localhost:6006"

docker-down:
	docker compose down

test:
	pytest tests/ -v --cov=agents --cov=api --cov=bias_db \
		--cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

analyze-demo:
	@echo "Running demo analysis on sample job description..."
	curl -s -X POST "http://localhost:8000/analyze" \
		-H "Content-Type: application/json" \
		-d "$$(cat data/inputs/sample_jd_biased.txt | python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read(), "doc_type": "job_description"}))')" \
		| python3 -m json.tool

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	@echo "✅ Cleaned"
