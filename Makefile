.PHONY: install lint test demo serve build

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/python -m pytest -q

demo:
	.venv/bin/python -m abeval compare examples/run_a.jsonl examples/run_b.jsonl
	.venv/bin/python -m abeval power --baseline 0.75 --delta 0.03

serve:
	.venv/bin/python -m http.server 8765 -d docs

# Build sdist + wheel from a clean `git archive` export in a temp directory
# OUTSIDE the working tree (hatchling walks upward and would otherwise pick up
# local ignore files), so only committed files can reach a distribution.
# The artifacts are audited afterwards.
build:
	rm -rf dist
	@export_dir=$$(mktemp -d) && \
	git archive HEAD | tar -x -C "$$export_dir" && \
	.venv/bin/python -m build --outdir dist "$$export_dir" && \
	rm -rf "$$export_dir"
	@if tar -tzf dist/*.tar.gz | grep -qiE 'claude|gitignore'; then \
		echo "TAINTED SDIST"; exit 1; fi
	@if unzip -l dist/*.whl | grep -qiE 'claude|gitignore'; then \
		echo "TAINTED WHEEL"; exit 1; fi
	@echo "artifacts clean:" && ls dist
