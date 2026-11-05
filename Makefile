# RiskCanvas — convenience targets.
# Usage: `make <target>`. Run `make help` (or just `make`) to list targets.

.DEFAULT_GOAL := help
.PHONY: help install test test-cov lint smoke capture api frontend frontend-build frontend-lint demo demo-local clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install backend package with dev extras, using the pinned set
	pip install -e ".[dev]" -c requirements.txt

test: ## Run the backend test suite
	pytest tests/

test-cov: ## Run backend tests with coverage report
	pytest tests/ --cov=riskcanvas --cov-report=term-missing -v

lint: ## Byte-compile the package as a basic sanity check
	python -m compileall riskcanvas

smoke: ## Live-fire driver: real API + real browser, asserts documented drawdowns
	node e2e/smoke.mjs

capture: ## Regenerate docs/screenshot.png + frontend/public/og.png (+ docs/capture.json) from the live stack
	node e2e/capture.mjs

api: ## Run the FastAPI service on http://localhost:8004
	uvicorn riskcanvas.api.app:app --reload --port 8004

frontend: ## Install deps and run the Next.js demo (http://localhost:3004)
	cd frontend && npm install && npm run dev

demo: ## One command: bring up the FULL stack via Docker (API :8004 + UI :3004)
	docker compose up --build

demo-local: ## Full stack without Docker: API on :8004 + Next dev on :3004
	@echo "Starting RiskCanvas API on :8004 and frontend on :3004 ..."
	@echo "  API:      http://localhost:8004"
	@echo "  Frontend: http://localhost:3004"
	uvicorn riskcanvas.api.app:app --port 8004 & \
		echo $$! > /tmp/riskcanvas-api.pid; \
		cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8004 npm run dev; \
		kill `cat /tmp/riskcanvas-api.pid` 2>/dev/null || true

frontend-build: ## Production build of the frontend
	cd frontend && npm install && npm run build

frontend-lint: ## Lint the frontend
	cd frontend && npm install && npm run lint

clean: ## Remove build/test caches
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
