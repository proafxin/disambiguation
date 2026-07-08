# /usr/bin/bash

set -e
uv run pre-commit run --all-files
uv run mypy .
