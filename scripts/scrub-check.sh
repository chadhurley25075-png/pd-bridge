#!/usr/bin/env bash
# fail on any personal path/address/secret in tracked files (CI-able privacy gate)
# Generic patterns live here. Site-specific ones (your own usernames, hostnames, password fragments)
# go in .scrub-private (gitignored), one extended-regex per line — never in this file.
set -uo pipefail
cd "$(dirname "$0")/.."
GENERIC='192\.168\.[0-9]|100\.(6[4-9]|[7-9][0-9]|1[0-2][0-9])\.[0-9]+\.[0-9]+|/Users/[a-z]|/home/[a-z]+-?[a-z]*/|gho_[A-Za-z0-9]{20}|ghp_[A-Za-z0-9]{20}|sk-[A-Za-z0-9]{20}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY'
PRIVATE=""
[ -f .scrub-private ] && PRIVATE="$(grep -vE '^\s*(#|$)' .scrub-private | paste -sd'|' -)"
PAT="$GENERIC"; [ -n "$PRIVATE" ] && PAT="$GENERIC|$PRIVATE"
bad=0
while IFS= read -r f; do
  hits=$(grep -nE "$PAT" "$f" | grep -v '10\.0\.[0-9]\.' | grep -vE '^[0-9]+:Copyright ' || true)
  if [ -n "$hits" ]; then echo "== $f"; echo "$hits"; bad=1; fi
done < <(git ls-files | grep -vE 'scripts/scrub-check\.sh')
[ $bad -eq 0 ] && echo "scrub-check: clean" || { echo "scrub-check: FAILED"; exit 1; }
