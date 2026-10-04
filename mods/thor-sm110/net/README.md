# net — 10GbE rail tuning + patched nvethernet for the Forecr Thor pair

Everything here runs at boot via `forecr-net-tune.service` on both boards
(thorc1 192.168.1.230 / thorc2 192.168.1.24), lanes `mgbe0_0..mgbe3_0`,
rail 10.0.N.1/.2.

## Files

- `net_tune.sh` — the live tune script (verbatim from the boards). Order:
  1. `nvethernet_ensure_fixed.sh` (swap patched module in if initrd loaded stock)
  2. jumbo MTU 9000 (driver clamps to 8966) + coalescing
     `rx-usecs 8 rx-frames 1 tx-usecs 32 tx-frames 16` + **RX ring 16384**
     with a 5-try verify loop (NetworkManager can race the link up mid-sequence)
  3. threaded NAPI per interface
  4. DMA deferred-unmap (DMA-FQ) per IOMMU group
  5. `jetson_clocks` + **explicit CPU `scaling_min_freq` pin** (see gotcha below)
  6. lane IRQ spread: lane N -> cpu N+1 via `/proc/irq/*/smp_affinity_list`
  7. GPU devfreq pin min=max: `gpu-gpc-0` 1575 MHz, `gpu-nvd-0` 1692 MHz
- `nvethernet_ensure_fixed.sh` — initrd loads the STOCK nvethernet early boot;
  this compares GNU build-ids and reloads the patched
  `/lib/modules/.../nvethernet.ko` if the loaded one differs.
- `nvethernet_rx_lost_irq.patch` — the driver patch (ether_linux.c): after
  `napi_complete()` under `rlock`, re-check for DMA completions that raced in
  between the last poll and the re-arm, closing a lost-IRQ window that stranded
  RX frames. (This is the "ISR rlock + RX re-check" fix; the earlier
  `nvethernet_vm_isr_lock.patch` name from round notes refers to the same
  rlock fix lineage — only this consolidated patch file exists on the boards.)

## Fixed module identity

- Installed: `/lib/modules/6.8.12-1021-tegra/updates/drivers/net/ethernet/nvidia/nvethernet/nvethernet.ko`
- GNU build-id: `08bf9206199de4dce9f1384a461946bffc45f1cc` (identical both boards)
- Stock backup kept on boards as `~/nvethernet.ko.stock-20261001`.
- Build: patch `drivers/net/ethernet/nvidia/nvethernet/ether_linux.c` in the
  L4T kernel source, `make M=drivers/net/ethernet/nvidia/nvethernet modules`,
  install to the `updates/` path above, `depmod -a`.

## Key gotchas (measured, do not rediscover)

1. **`jetson_clocks` does NOT pin CPU frequency minimums at boot** — scaling_min
   stays at 972 MHz (thorc2 came up 972 vs thorc1 2601 once; in TP=2 the slow
   rank gates all 86 collectives/step). Hence step 5's explicit
   `scaling_min_freq = max_freq` loop. MAXN power mode != max clocks.
2. **RX ring 4096 silently overflows under decode bursts** -> 1-2.5 s TCP RTO
   stalls on ~13% of steps -> decode 8-16 tok/s. 16384 (hw max) restored
   45.6 tok/s. Invisible at every other layer (no NIC drop counters, no softnet
   drops) — only `nstat TcpExtTCPTimeouts` deltas + stall probes caught it.
3. **`rx-frames 1`** avoids stranded RX frames (stock coalescing stalls NCCL).
4. **GPU devfreq min must be pinned explicitly** (step 7) — thorc1 booted with
   gpc min=315 MHz after a reboot even with jetson_clocks.
5. MTU resets to 1500 every reboot — the tune service handles it; verify with
   `ping -M do -s 8972 <peer>` before serving.
6. nvethernet has no flow control (`ethtool -A` unsupported) and no DOM
   (`ethtool -m` unsupported).
7. 4-NIC striping through NCCL gives NO gain (per-flow hash + reordering);
   single-lane mgbe0_0 for NCCL, lanex uses all 4 lanes itself.

## Systemd units (as installed on the boards)

`/etc/systemd/system/forecr-net-tune.service`:
```
[Unit]
Description=Forecr Thor 10GbE rail tuning (MTU/coalescing/ring/IRQ/devfreq)
After=network-pre.target
Before=network-online.target docker.service

[Service]
Type=oneshot
ExecStart=/home/nebq29/net_tune.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

`jetson-clocks.service` (CPU pin companion, invoked from net_tune.sh step 5;
kept separate on the boards for manual re-run):
```
[Unit]
Description=Pin CPU/GPU clocks to max (jetson_clocks + scaling_min)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'jetson_clocks; for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq; do mx=$(cat ${c%min_freq}max_freq 2>/dev/null); [ -n "$mx" ] && echo $mx > $c; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

The NVIDIA driver-side analysis of the lost-IRQ bug is in the user's
`~/claude/NVIDIA_nvethernet_report.md` (not in this repo; ask the user to
copy it over if it should be tracked).
