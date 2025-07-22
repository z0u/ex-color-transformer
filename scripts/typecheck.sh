#!/usr/bin/env bash

set -euo pipefail

( set -x; uv run --no-sync basedpyright "$@" )

echo "✅ Type check passed"
