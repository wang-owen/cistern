#!/usr/bin/env bash
# Capture the environment of a benchmark run as JSON (env.json), using jq.
# usage: env.sh --target DIR [--idle-s N] [--out FILE]
# Records drive model, filesystem, free space, RAM, CPU, governor, kernel, ulimit -l, git SHA +
# dirty flag, idle time used. Reads only; needs no root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET="" IDLE_S=0 OUT=""
while (($#)); do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --idle-s) IDLE_S="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --dry-run) shift ;; # reads only; accepted for uniformity
        -h | --help) sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$TARGET" ]] || { echo "--target is required" >&2; exit 2; }
[[ -d "$TARGET" ]] || { echo "target is not a directory: $TARGET" >&2; exit 2; }

# --- drives from sysfs (no root) ---
drives='[]'
for d in /sys/class/nvme/nvme*; do
    [[ -d "$d" ]] || continue
    rd() { [[ -r "$d/$1" ]] && tr -d '\n' <"$d/$1" || true; }
    drives="$(jq -c --arg dev "$(basename "$d")" --arg model "$(rd model)" \
        --arg fw "$(rd firmware_rev)" --arg transport "$(rd transport)" --arg state "$(rd state)" \
        '. + [{dev:$dev, model:($model|sub("\\s+$";"")), firmware:$fw, transport:$transport, state:$state}]' <<<"$drives")"
done

# nvme list -o json if readable; kept only if it is valid JSON
nvme_list=null
if command -v nvme >/dev/null 2>&1; then
    if out="$(nvme list -o json 2>/dev/null)" && jq -e . >/dev/null 2>&1 <<<"$out"; then
        nvme_list="$out"
    fi
fi

# --- filesystem ---
fs_json="$(findmnt -T "$TARGET" -J -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null | jq -c '.filesystems[0] // null' || echo null)"
[[ -n "$fs_json" ]] || fs_json=null
read -r fs_size fs_avail < <(df -B1 --output=size,avail "$TARGET" | tail -n1)
src="$(findmnt -T "$TARGET" -no SOURCE 2>/dev/null || true)"
sched=null
if [[ -n "$src" && -e "$src" ]]; then
    parent="$(lsblk -no PKNAME "$src" 2>/dev/null | head -n1 || true)"
    blk="${parent:-$(basename "$src")}"
    if [[ -r "/sys/block/$blk/queue/scheduler" ]]; then
        sched="$(jq -n --arg s "$(cat "/sys/block/$blk/queue/scheduler")" '$s')"
    fi
fi

# --- host ---
mem_total="$(awk '/^MemTotal:/ {print $2*1024}' /proc/meminfo)"
mem_avail="$(awk '/^MemAvailable:/ {print $2*1024}' /proc/meminfo)"
lscpu_json="$(lscpu -J 2>/dev/null | jq -c . || echo null)"
gov_file=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
governor="unavailable"
[[ -r "$gov_file" ]] && governor="$(cat "$gov_file")"
ulimit_l="$(ulimit -l)"

sha="nogit" dirty=false
if git -C "$ROOT" rev-parse --short=7 HEAD >/dev/null 2>&1; then
    sha="$(git -C "$ROOT" rev-parse --short=7 HEAD)"
fi
if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then dirty=true; fi

json="$(jq -n \
    --argjson drives "$drives" --argjson nvme_list "$nvme_list" \
    --argjson fs "$fs_json" --arg target "$TARGET" \
    --argjson fs_size "$fs_size" --argjson fs_avail "$fs_avail" --argjson sched "$sched" \
    --argjson mem_total "$mem_total" --argjson mem_avail "$mem_avail" \
    --argjson lscpu "$lscpu_json" --arg governor "$governor" \
    --arg kernel "$(uname -r)" --arg ulimit_l "$ulimit_l" \
    --arg sha "$sha" --argjson dirty "$dirty" --argjson idle "$IDLE_S" \
    --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg host "$(uname -m)" \
    '{schema:"cistern.env/1", synthetic:false, ts:$ts, target:$target,
      drives:$drives, nvme_list:$nvme_list,
      filesystem:{mount:$fs, size_bytes:$fs_size, free_bytes:$fs_avail, io_scheduler:$sched},
      ram:{total_bytes:$mem_total, available_bytes:$mem_avail},
      cpu:$lscpu, arch:$host, governor:$governor, kernel:$kernel,
      ulimit_l:$ulimit_l, git_sha:$sha, git_dirty:$dirty, idle_s:$idle}')"

if [[ -n "$OUT" ]]; then
    printf '%s\n' "$json" >"$OUT"
else
    printf '%s\n' "$json"
fi
