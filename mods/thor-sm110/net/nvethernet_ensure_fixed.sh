#!/bin/bash
# The initrd loads the stock nvethernet at early boot. Swap in the patched
# /lib/modules copy (ISR rlock fix + RX re-check) if a different one is loaded.
# Compares GNU build-ids; if either can't be read, it reloads anyway.
f=$(modinfo -n nvethernet 2>/dev/null)
want=$(readelf -n "$f" 2>/dev/null | awk '/Build ID/{print $3}')
n=/sys/module/nvethernet/notes/.note.gnu.build-id
have=$([ -r $n ] && od -An -tx1 -v $n | tr -d ' \n' | tail -c 40)
if [ -n "$want" ] && [ "$want" = "$have" ]; then
  echo "nvethernet-ensure: fixed module already loaded ($have)"; exit 0
fi
echo "nvethernet-ensure: loaded=${have:-?} want=${want:-?}, reloading"
coe=0; lsmod | grep -q '^tegra_capture_coe' && coe=1
[ $coe = 1 ] && rmmod tegra_capture_coe
if ! rmmod nvethernet; then
  echo "nvethernet-ensure: rmmod failed, leaving stock loaded"
  [ $coe = 1 ] && modprobe tegra_capture_coe; exit 0
fi
modprobe nvethernet
[ $coe = 1 ] && modprobe tegra_capture_coe
for s in $(seq 1 30); do
  c=0; for k in 0 1 2 3; do ip -4 addr show dev mgbe${k}_0 2>/dev/null | grep -q 'inet 10\.0\.' && c=$((c+1)); done
  [ $c = 4 ] && break; sleep 1
done
have=$([ -r $n ] && od -An -tx1 -v $n | tr -d ' \n' | tail -c 40)
echo "nvethernet-ensure: now loaded=${have:-?} lanes_with_addr=$c"
