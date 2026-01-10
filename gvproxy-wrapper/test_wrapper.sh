#!/bin/bash
set -e

echo "=== Testing gvproxy-wrapper with DNS zones ==="

# Create DNS zones JSON
DNS_ZONES='[
  {
    "name": "allowed",
    "records": [
      {
        "name": "pypi.org",
        "regex": "^.*\\.pypi\\.org$",
        "ip": "1.2.3.4"
      },
      {
        "name": "files.pythonhosted.org",
        "regex": "^.*\\.files\\.pythonhosted\\.org$",
        "ip": "1.2.3.4"
      }
    ],
    "defaultIP": "0.0.0.0"
  }
]'

echo "DNS Zones configuration:"
echo "$DNS_ZONES" | python3 -m json.tool

echo ""
echo "Starting gvproxy-wrapper..."
./gvproxy-wrapper-linux-arm64 \
  -listen-qemu unix:///tmp/test-wrapper.sock \
  -dns-zones "$DNS_ZONES" \
  -debug &

WRAPPER_PID=$!
echo "Wrapper PID: $WRAPPER_PID"

# Wait for socket
sleep 3

if [ -S /tmp/test-wrapper.sock ]; then
    echo "✅ Wrapper created socket successfully"
    kill $WRAPPER_PID 2>/dev/null || true
    rm -f /tmp/test-wrapper.sock
    echo "✅ Test passed"
else
    echo "❌ Socket not created"
    kill $WRAPPER_PID 2>/dev/null || true
    exit 1
fi
