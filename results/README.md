# results/

```
raw/<experiment>/   tool output exactly as produced (fio text + bw logs, df before/after, ...)
raw/environment/    env.json + env.txt from the machine, captured once per session
processed/          generated from raw/ by the plotting script; never edited by hand
```

Every result directory should carry the environment it was measured in:

```bash
tools/env.sh --target /mnt/nvme --idle-s 30 --out results/raw/<experiment>/env.json
```

`env.json` (schema `cistern.env/2`) records drive, filesystem and free space, RAM, CPU, governor,
kernel, `ulimit -l`, git SHA and dirty flag, idle time, and (if present) the AMD GPUs: name and
arch, ROCm version, VRAM, clock state, and the PCIe link of the GPU **and every bridge above it**.
The GPU's own link entry can read x16 at 32 GT/s while the root port runs slower; the slot's real
bandwidth is the minimum along the chain.

Rules (from the blueprint): 3 runs with a discarded warmup, median with min/max, differences within
spread are "no change", idle (plus `fstrim`) between runs, idle machine.
