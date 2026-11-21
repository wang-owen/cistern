#!/usr/bin/env bash
# Run the QD16 fio ceiling (write then read) and record everything in
# results/raw/fio_ceiling_qd16/: fio.txt (both jobs), write_bw.1.log, read_bw.1.log,
# df_before.txt, df_after.txt, env.json.
# usage: FIO_FILE=/mnt/nvme/fio.tmp [IDLE_S=30] tools/fio/run_ceiling.sh
# Does not fstrim or idle for you: do that first, as in the drift check. Removes the fio file after.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIO_FILE="${FIO_FILE:-/mnt/nvme/fio.tmp}"
IDLE_S="${IDLE_S:-0}"
OUT="$ROOT/results/raw/fio_ceiling_qd16"
JOB="$ROOT/tools/fio/ceiling_qd16.fio"
export FIO_FILE

mkdir -p "$OUT"
TARGET_DIR="$(dirname "$FIO_FILE")"
[[ -d "$TARGET_DIR" ]] || { echo "not a directory: $TARGET_DIR" >&2; exit 2; }

df -hT "$TARGET_DIR" >"$OUT/df_before.txt"
"$ROOT/tools/env.sh" --target "$TARGET_DIR" --idle-s "$IDLE_S" --out "$OUT/env.json"

# fio writes the bw logs into the working directory; the jobs name them write_bw.1.log / read_bw.1.log
cd "$OUT"
fio --output="$OUT/fio.txt" "$JOB"

df -hT "$TARGET_DIR" >"$OUT/df_after.txt"
rm -f "$FIO_FILE"
echo "results in $OUT"
