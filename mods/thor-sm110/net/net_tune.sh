#!/bin/bash
/home/nebq29/nvethernet_ensure_fixed.sh
# Forecr Thor pair QSFP 4x10GbE optimization per NVIDIA R39.2.1 guide
IFACES="mgbe0_0 mgbe1_0 mgbe2_0 mgbe3_0"

# 1. Jumbo MTU (driver clamps 9000->8966) + coalescing latency fix
for i in $IFACES; do
  for try in 1 2 3 4 5; do   # NM can bring the link up mid-sequence; verify and retry
    sudo ip link set dev $i down
    sudo ip link set dev $i mtu 9000
    sudo ethtool -C $i rx-usecs 8 rx-frames 1 tx-usecs 32 tx-frames 16  # rx-frames 1 avoids stranded RX frames (NCCL stalls), 2026-10-01
    sudo ethtool -G $i rx 16384 tx 4096
    sudo ip link set dev $i up
    m=$(cat /sys/class/net/$i/mtu); f=$(ethtool -c $i | awk '/^rx-frames:/{print $2}')
    [ "$m" = 8966 ] && [ "$f" = 1 ] && break
    echo "$i: try $try got mtu=$m rx-frames=$f, retrying"; sleep 2
  done
done

# 2. Threaded NAPI
for i in $IFACES; do
  ETH=$(readlink -f /sys/class/net/$i/device | grep -oE "[0-9a-f]+\.ethernet$")
  if [ -n "$ETH" ]; then
    echo 1 | sudo tee /sys/devices/platform/bus@0/$ETH/net/$i/threaded
  else
    echo "NAPI path not found for $i"
  fi
done

# 3. Deferred DMA unmap (DMA-FQ) per IOMMU group
for i in $IFACES; do
  GROUP=$(basename $(readlink /sys/class/net/$i/device/iommu_group))
  echo DMA-FQ | sudo tee /sys/kernel/iommu_groups/$GROUP/type
done

# 4. Max CPU/GPU clocks
sudo jetson_clocks || echo "jetson_clocks failed"
# 4b. EXPLICIT CPU clock pin (jetson_clocks alone does NOT raise scaling_min at boot)
for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq; do
  mx="$(cat "${c%min_freq}max_freq" 2>/dev/null)"
  if [ -n "$mx" ]; then echo "$mx" > "$c" 2>/dev/null; fi
done


ip -br addr show | grep mgbe
for i in $IFACES; do echo -n "$i mtu: "; cat /sys/class/net/$i/mtu; done

# Round 7 (Oct 2026): spread lane IRQs off cpu0 — one lane per CPU (lane N -> cpu N+1).
# Evidence: 16 mgbe IRQs on cpu0 = 60k+ irq/s during 18MB exchanges, cpu0 63% busy.
# Spread cut lanex xchg 6.9ms -> 5.6ms (3.18 GB/s, near bare-transport 3.6). Neutral for MTP-on decode.
for i in 0 1 2 3; do
  for n in $(grep "mgbe${i}_0\." /proc/interrupts | cut -d: -f1); do
    echo $((i+1)) > /proc/irq/$n/smp_affinity_list 2>/dev/null
  done
done

# Round 7 (2026-10-02): spread 10GbE lane IRQs off cpu0 (lane N -> cpu N+1)
for i in 0 1 2 3; do
  for n in $(grep "mgbe${i}_0\." /proc/interrupts | cut -d: -f1); do
    echo $((i+1)) > /proc/irq/$n/smp_affinity_list 2>/dev/null
  done
done

# Round 9 (Oct 3): pin GPU devfreq min=max at boot (jetson_clocks alone leaves min at 315MHz
# on thorc1; round-9 requires GPU min=max). gpc 1575MHz, nvd 1692MHz.
echo 1575000000 > /sys/class/devfreq/gpu-gpc-0/min_freq 2>/dev/null
echo 1692000000 > /sys/class/devfreq/gpu-nvd-0/min_freq 2>/dev/null
