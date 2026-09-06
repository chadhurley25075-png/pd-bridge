#!/usr/bin/env bash
# Build the HF hub layout vLLM expects, with HARD LINKS inside the mounted filesystem (symlinks out of the mount are invisible in the container).
set -uo pipefail
: "${SRC:?set SRC=<dir holding the DeepSeek-V4-Flash FP8 checkpoint>}"
: "${HF_DIR:?set HF_DIR}"
HUB=$HF_DIR/hub
D=$HUB/models--deepseek-ai--DeepSeek-V4-Flash; mkdir -p $D/snapshots/main $D/refs
n=0; for f in "$SRC"/*; do b=$(basename "$f"); [ -e "$D/snapshots/main/$b" ] || ln "$f" "$D/snapshots/main/$b" 2>/dev/null || cp "$f" "$D/snapshots/main/$b"; n=$((n+1)); done
printf main > $D/refs/main
echo "$(hostname): $n files linked -> $D/snapshots/main ($(du -sh $D/snapshots/main | cut -f1)); config: $(ls $D/snapshots/main/config.json)"
