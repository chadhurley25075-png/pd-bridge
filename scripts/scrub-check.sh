#!/usr/bin/env bash
# fail on any personal path/address/secret in tracked files (CI-able privacy gate)
set -uo pipefail
cd "$(dirname "$0")/.."
bad=0
while IFS= read -r f; do
  hits=$(grep -nE '192\.168\.[0-9]|100\.[0-9]+\.[0-9]+\.[0-9]+|/Users/[a-z]|/home/[a-z]+-?[a-z]*/|P@ss|USER|Hurley' "$f" \
    | grep -v '10\.0\.0\.' | grep -vE '^[0-9]+:Copyright ' || true)
  if [ -n "$hits" ]; then echo "== $f"; echo "$hits"; bad=1; fi
done < <(git ls-files | grep -vE '\.md$|config\.example\.env|scripts/scrub-check\.sh' ; git ls-files '*.md' | grep -v RESULTS.md)
# config.example.env is allowed illustrative RFC1918 addresses only if labeled; keep it strict anyway:
grep -nE 'P@ss|USER|Hurley|/Users/[a-z]' config.example.env && bad=1
[ $bad -eq 0 ] && echo "scrub-check: clean" || { echo "scrub-check: FAILED"; exit 1; }
