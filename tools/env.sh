#!/usr/bin/env bash
# Capture the environment of a benchmark run as JSON (env.json), using jq.
# usage: env.sh --target DIR [--idle-s N] [--out FILE]
# Records drive model, filesystem, free space, RAM, CPU, governor, kernel, ulimit -l, git SHA +
# dirty flag, idle time used, plus (when present) AMD GPUs: name/arch, ROCm version, VRAM, clock
# state, and the PCIe link of the GPU and every bridge above it. Reads only; needs no root.
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

# --- GPUs (AMD, via sysfs; empty array if none) ---
# The link of the device alone is not the slot link: Navi parts sit behind an internal PCIe
# switch, so every PCI function from the root port down is recorded.
pci_link() { # $1 = /sys/bus/pci/devices/<addr> ; prints one JSON object
    local d="$1" f
    rdp() { [[ -r "$d/$1" ]] && tr -d '\n' <"$d/$1" || true; }
    jq -nc --arg addr "$(basename "$d")" --arg cs "$(rdp current_link_speed)" \
        --arg cw "$(rdp current_link_width)" --arg ms "$(rdp max_link_speed)" \
        --arg mw "$(rdp max_link_width)" \
        '{addr:$addr, current_speed:$cs, current_width:$cw, max_speed:$ms, max_width:$mw}'
}
gpus='[]'
for c in /sys/class/drm/card*; do
    [[ "$(basename "$c")" =~ ^card[0-9]+$ ]] || continue
    dev="$c/device"
    [[ -r "$dev/vendor" && "$(cat "$dev/vendor")" == 0x1002 ]] || continue
    real="$(readlink -f "$dev")"
    chain='[]'
    cur="$real"
    # walk from the device up through its PCI bridges (stop at the pciDDDD:BB host bridge dir)
    while [[ "$(basename "$cur")" =~ ^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$ ]]; do
        chain="$(jq -c --argjson l "$(pci_link "$cur")" '. + [$l]' <<<"$chain")"
        cur="$(dirname "$cur")"
    done
    rdg() { [[ -r "$dev/$1" ]] && tr -d '\n' <"$dev/$1" || true; }
    rdg_raw() { [[ -r "$dev/$1" ]] && cat "$dev/$1" || true; }
    gpus="$(jq -c --arg card "$(basename "$c")" --arg addr "$(basename "$real")" \
        --arg vt "$(rdg mem_info_vram_total)" --arg vu "$(rdg mem_info_vram_used)" \
        --arg lvl "$(rdg power_dpm_force_performance_level)" \
        --arg sclk "$(rdg_raw pp_dpm_sclk)" --arg mclk "$(rdg_raw pp_dpm_mclk)" \
        --argjson chain "$chain" \
        '. + [{card:$card, pci_addr:$addr,
               vram_total_bytes:(if $vt=="" then null else ($vt|tonumber) end),
               vram_used_bytes:(if $vu=="" then null else ($vu|tonumber) end),
               perf_level:$lvl, pp_dpm_sclk:$sclk, pp_dpm_mclk:$mclk,
               pcie_chain_device_to_root:$chain}]' <<<"$gpus")"
done
# GPU agents as ROCm sees them (name + gfx arch); null when rocminfo is missing
rocm_agents=null
if command -v rocminfo >/dev/null 2>&1; then
    # per agent, rocminfo prints "Name:" (gfx id for GPUs) before "Marketing Name:"
    rocm_agents="$(rocminfo 2>/dev/null | awk '
        /^[ \t]+Name:[ \t]+/ { gfx = ($2 ~ /^gfx/) ? $2 : "" }
        /Marketing Name:/ { if (gfx != "") { sub(/^[^:]*:[ \t]*/, ""); sub(/[ \t]+$/, ""); printf "%s\t%s\n", gfx, $0; gfx = "" } }' |
        jq -Rsc 'split("\n") | map(select(length>0) | split("\t") | {gfx:.[0], name:.[1]})' || echo null)"
    [[ -n "$rocm_agents" ]] || rocm_agents=null
fi
rocm_version=""
if command -v hipconfig >/dev/null 2>&1; then rocm_version="$(hipconfig --version 2>/dev/null || true)"; fi

# --- toolchain ---
liburing_ver="$(pkg-config --modversion liburing 2>/dev/null || true)"
gxx_ver="$(g++ --version 2>/dev/null | head -n1 || true)"
meson_ver="$(meson --version 2>/dev/null || true)"

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
    --argjson gpus "$gpus" --argjson rocm_agents "$rocm_agents" --arg rocm_version "$rocm_version" \
    --arg liburing "$liburing_ver" --arg gxx "$gxx_ver" --arg meson "$meson_ver" \
    '{schema:"cistern.env/2", synthetic:false, ts:$ts, target:$target,
      drives:$drives, nvme_list:$nvme_list,
      filesystem:{mount:$fs, size_bytes:$fs_size, free_bytes:$fs_avail, io_scheduler:$sched},
      ram:{total_bytes:$mem_total, available_bytes:$mem_avail},
      cpu:$lscpu, arch:$host, governor:$governor, kernel:$kernel,
      gpu:{rocm_version:$rocm_version, rocm_agents:$rocm_agents, devices:$gpus},
      toolchain:{liburing:$liburing, gxx:$gxx, meson:$meson},
      ulimit_l:$ulimit_l, git_sha:$sha, git_dirty:$dirty, idle_s:$idle}')"

if [[ -n "$OUT" ]]; then
    printf '%s\n' "$json" >"$OUT"
else
    printf '%s\n' "$json"
fi
