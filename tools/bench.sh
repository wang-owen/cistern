#!/usr/bin/env bash
# The two benchmarks, under the same conditions: the fio write ceiling (8 GiB sequential write,
# 1 MiB blocks, QD16, io_uring, O_DIRECT, fixed buffers, fallocate) and checkpoint write
# throughput (cistern-bench-write). Each round runs fio, then the writer, each after IDLE_S of
# idle (and fstrim when passwordless sudo works). Round 0 is a discarded warmup.
# Writes results/ceiling.txt and results/writer.txt (GB/s per round, warmup first and marked) and
# results/env.json (machine, drive and git state). Report the median with min/max; differences
# within that spread are "no change".
# usage: ninja -C build-release && [TARGET=/mnt/nvme] [IDLE_S=30] [RUNS=3] tools/bench.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-/mnt/nvme}"
IDLE_S="${IDLE_S:-30}"
RUNS="${RUNS:-3}"
WRITER="$ROOT/build-release/tools/cistern-bench-write"
OUT="$ROOT/results"
FIO_FILE="$TARGET/fio.tmp"
WDIR="$TARGET/cistern-bench"
[[ -x "$WRITER" ]] || { echo "build first: ninja -C build-release" >&2; exit 2; }
[[ -d "$TARGET" ]] || { echo "not a directory: $TARGET" >&2; exit 2; }
trim=false; sudo -n fstrim --version >/dev/null 2>&1 && trim=true

# Environment first: writing results/ would mark the tree dirty.
src="$(findmnt -T "$TARGET" -no SOURCE)"
blk="$(lsblk -no PKNAME "$src" 2>/dev/null | head -n1 || true)"
blk="${blk:-$(basename "$src")}"
sysfs() { [[ -r "/sys/block/$blk/$1" ]] && sed 's/ *$//' "/sys/block/$blk/$1" || true; }
read -r fs_size fs_free < <(df -B1 --output=size,avail "$TARGET" | tail -n1)
dirty=false; [[ -z "$(git -C "$ROOT" status --porcelain)" ]] || dirty=true
env_json="$(jq -n \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg target "$TARGET" \
    --arg model "$(sysfs device/model)" --arg firmware "$(sysfs device/firmware_rev)" \
    --arg sched "$(sysfs queue/scheduler)" \
    --arg fstype "$(findmnt -T "$TARGET" -no FSTYPE)" \
    --arg fsopts "$(findmnt -T "$TARGET" -no OPTIONS)" \
    --argjson fs_size "$fs_size" --argjson fs_free "$fs_free" \
    --argjson mem_total "$(awk '/^MemTotal:/ {print $2*1024}' /proc/meminfo)" \
    --argjson mem_avail "$(awk '/^MemAvailable:/ {print $2*1024}' /proc/meminfo)" \
    --arg cpu "$(lscpu | sed -n 's/^Model name: *//p')" \
    --arg governor "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)" \
    --arg kernel "$(uname -r)" --arg ulimit_l "$(ulimit -l)" \
    --arg liburing "$(pkg-config --modversion liburing)" --arg gxx "$(g++ --version | head -n1)" \
    --arg fio "$(fio --version)" --arg sha "$(git -C "$ROOT" rev-parse --short=7 HEAD)" \
    --argjson dirty "$dirty" --argjson idle_s "$IDLE_S" --argjson fstrim "$trim" \
    '{ts:$ts, target:$target, drive:{model:$model, firmware:$firmware, io_scheduler:$sched},
      filesystem:{type:$fstype, options:$fsopts, size_bytes:$fs_size, free_bytes:$fs_free},
      ram:{total_bytes:$mem_total, available_bytes:$mem_avail}, cpu:$cpu, governor:$governor,
      kernel:$kernel, ulimit_l:$ulimit_l, liburing:$liburing, gxx:$gxx, fio:$fio,
      git_sha:$sha, git_dirty:$dirty, idle_s:$idle_s, fstrim:$fstrim}')"
mkdir -p "$OUT"
printf '%s\n' "$env_json" >"$OUT/env.json"
: >"$OUT/ceiling.txt"
: >"$OUT/writer.txt"

settle() {
    rm -rf "$WDIR" "$FIO_FILE"; mkdir -p "$WDIR"; sync
    if $trim; then sudo -n fstrim "$TARGET"; fi
    sleep "$IDLE_S"
}
for ((r = 0; r <= RUNS; r++)); do
    tag=""; ((r == 0)) && tag=" warmup"
    settle
    ceiling="$(fio --name=write --filename="$FIO_FILE" --rw=write --bs=1M --size=8G \
        --ioengine=io_uring --iodepth=16 --direct=1 --fixedbufs=1 --fallocate=native \
        --output-format=json | jq '.jobs[0].write.bw_bytes / 1e9')"
    printf '%.3f%s\n' "$ceiling" "$tag" >>"$OUT/ceiling.txt"
    settle
    writer="$("$WRITER" "$WDIR")"
    printf '%s%s\n' "$writer" "$tag" >>"$OUT/writer.txt"
    printf 'round %d: ceiling %.3f  writer %s\n' "$r" "$ceiling" "$writer"
done
rm -rf "$WDIR" "$FIO_FILE"
