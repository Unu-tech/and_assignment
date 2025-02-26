SHELL := /bin/bash

.DEFAULT_GOAL := help
.PHONY: install check pylint pyright style black isort help

install: ## Install all dependencies
		pip install -r requirements.txt

check: pylint pyright ## Run pylint and pyright

pylint: ## Check code smells with pylint
		python -m pylint --exit-zero src

pyright: ## Check type annotations
		python -m pyright

style: black isort ## Run black and isort

black: ## Auto-format python code using black
		python -m black src

isort: ## Auto-format python code using isort
		python -m isort src

help: # Run `make help` to get help on the make commands
		@echo "\033[36mAvailable commands:\033[0m"
		@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
