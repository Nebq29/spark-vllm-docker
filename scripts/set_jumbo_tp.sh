#!/bin/bash
# Set persistent jumbo MTU (8966) on the DSv4 TP rail mgbe0_0 via NetworkManager.
# Survives reboot. Run on BOTH thorc1 (.24) and thorc2 (.230).
# NOTE: mgbe0_0 carries the 10.0.0.x cluster net; SSH sessions ride WiFi (wlP1p1s0),
# so a brief link flap here does NOT cut us off.
set -e
IDX="$1"   # 1 => 10.0.0.1 (node1/thorc1), 2 => 10.0.0.2 (node2/thorc2)

echo "=== before ==="
nmcli -f connection.interface-name,802-3-ethernet.mtu con show mgbe0 2>/dev/null || true
cat /sys/class/net/mgbe0_0/mtu

sudo nmcli con mod mgbe0 802-3-ethernet.mtu 8966
sudo nmcli con up mgbe0 2>/dev/null || {
  # if 'up' complains, force cycle the link
  sudo ip link set dev mgbe0_0 down; sleep 1; sudo ip link set dev mgbe0_0 up
  sleep 2
}
# ensure the cluster IP is present (noprefixroute static)
if ! ip -4 addr show dev mgbe0_0 | grep -q "10.0.0.${IDX}/24"; then
  sudo ip addr add 10.0.0.${IDX}/24 dev mgbe0_0 2>/dev/null || true
fi

echo "=== after ==="
echo "mtu=$(cat /sys/class/net/mgbe0_0/mtu)"
ip -4 addr show dev mgbe0_0 | grep inet
echo "=== connectivity check to peer ==="
PEER=10.0.0.$([ "$IDX" = "1" ] && echo 2 || echo 1)
ping -c 3 -M do -s 8938 $PEER 2>&1 | tail -2
