.PHONY: help lint format check ci clean install-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install-dev: ## Install development dependencies
	pip install ruff pre-commit

lint: ## Run ruff linting
	ruff check .

format: ## Format code with ruff
	ruff format .

fix: ## Fix auto-fixable linting issues
	ruff check . --fix

check: ## Run all CI checks
	./scripts/ci_checks.sh

ci: ## Run complete CI pipeline (lint + check)
	ruff check .
	./scripts/ci_checks.sh

pre-commit: ## Run pre-commit checks
	./scripts/pre-commit-hook.sh

clean: ## Clean up Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

test-lint: ## Test that linting configuration works
	@echo "Testing ruff configuration..."
	ruff check . --output-format=text | head -20
	@echo "✅ Ruff configuration test completed"

test-ci: ## Test that CI checks work
	@echo "Testing CI checks..."
	./scripts/ci_checks.sh
	@echo "✅ CI checks test completed"

test-all: ## Run all tests
	make test-lint
	make test-ci
	@echo "🎉 All tests passed!"

setup-hooks: ## Set up git pre-commit hooks
	@echo "Setting up git pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

dev-setup: ## Complete development environment setup
	make install-dev
	make setup-hooks
	@echo "🎉 Development environment setup complete!"
