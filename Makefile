POETRY ?= poetry
SPHINX ?= sphinx-build

.PHONY: check.lock
check.lock:
	@$(POETRY) check --lock

.PHONY: clean
clean: clean.doc clean.py clean.test

.PHONY: clean.doc
clean.doc:
	@$(RM) -rf site/
	@$(RM) -rf docs/_build/
	@$(RM) -rf docs/auto_examples/
	@$(RM) -rf docs/sg_execution_times.rst

.PHONY: clean.py
clean.py:
	@find . -name __pycache__ -exec $(RM) -r {} +

.PHONY: clean.test
clean.test:
	@$(RM) -r .pytest_cache/

.PHONY: doc
doc: clean.doc install.doc
	@$(SPHINX) -b html docs/ docs/_build/html

.PHONY: install
install: check.lock
	@$(POETRY) install

.PHONY: install.dev
install.dev: check.lock
	@$(POETRY) install --only dev

.PHONY: install.doc
install.doc: check.lock
	@$(POETRY) install --only docs

.PHONY: lock
lock:
	@$(POETRY) lock --no-update

.PHONY: test
test: install
	@$(POETRY) run python -m pytest -v tests

.PHONY: test.cov
test.cov: install
	coverage run --source=src -m pytest -v tests --junitxml=report.xml
	
.PHONY: cov
cov: test.cov
	coverage report
