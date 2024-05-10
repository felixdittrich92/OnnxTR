.PHONY: quality style test  docs-single-version docs
# this target runs checks on all files
quality:
	ruff check .
	mypy onnxtr/

# this target runs checks on all files and potentially modifies some of them
style:
	ruff format .
	ruff check --fix .

# Run tests for the library
test:
	coverage run -m pytest tests/common/ -rs
	coverage report --fail-under=80 --show-missing

# Check that docs can build
docs-single-version:
	sphinx-build docs/source docs/_build -a

# Check that docs can build
docs:
	cd docs && bash build.sh