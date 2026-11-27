#!/usr/bin/env bash
# Writer vs. baseline vs. fio ceiling. Runs 1 discarded warmup + RUNS measured rounds; each round
# runs cistern-bench-write then cistern-baseline --mode odirect, with IDLE_S of idle (and fstrim
# when passwordless sudo is available) before every run. Records into results/raw/writer_qd16/:
# writer.txt, baseline_odirect.txt (one GB/s per line, warmup first, marked), writer_stderr.txt,
# df_before.txt, df_after.txt, env.json, trim.txt.
# usage: [TARGET=/mnt/nvme] [IDLE_S=30] [RUNS=3] [BUILD=build-release] tools/bench/run_writer.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${TARGET:-/mnt/nvme}"
IDLE_S="${IDLE_S:-30}"
RUNS="${RUNS:-3}"
BUILD="$ROOT/${BUILD:-build-release}"
OUT="$ROOT/results/raw/writer_qd16"
WRITER="$BUILD/bench/harness/cistern-bench-write"
BASELINE="$BUILD/bench/baselines/cistern-baseline"
[[ -x "$WRITER" && -x "$BASELINE" ]] || { echo "build first: ninja -C $BUILD" >&2; exit 2; }
[[ -d "$TARGET" ]] || { echo "not a directory: $TARGET" >&2; exit 2; }

mkdir -p "$OUT"
"$ROOT/tools/env.sh" --target "$TARGET" --idle-s "$IDLE_S" --out "$OUT/env.json"
df -hT "$TARGET" >"$OUT/df_before.txt"
: >"$OUT/writer.txt"; : >"$OUT/baseline_odirect.txt"; : >"$OUT/writer_stderr.txt"
if sudo -n fstrim --version >/dev/null 2>&1; then trim=1; echo "fstrim before every run" >"$OUT/trim.txt"
else trim=0; echo "no fstrim (no passwordless sudo); idle only" >"$OUT/trim.txt"; fi

WDIR="$TARGET/cistern-bench"
settle() {
    rm -rf "$WDIR" "$TARGET/baseline.bin"; mkdir -p "$WDIR"; sync
    ((trim)) && sudo -n fstrim "$TARGET"
    sleep "$IDLE_S"
}
for ((r = 0; r <= RUNS; r++)); do
    tag=""; ((r == 0)) && tag=" warmup"
    settle; echo "$("$WRITER" "$WDIR" 2>>"$OUT/writer_stderr.txt")$tag" >>"$OUT/writer.txt"
    settle; echo "$("$BASELINE" --mode odirect "$TARGET/baseline.bin")$tag" >>"$OUT/baseline_odirect.txt"
    echo "round $r: writer $(tail -1 "$OUT/writer.txt")  baseline $(tail -1 "$OUT/baseline_odirect.txt")"
done
rm -rf "$WDIR" "$TARGET/baseline.bin"
df -hT "$TARGET" >"$OUT/df_after.txt"
echo "results in $OUT"
