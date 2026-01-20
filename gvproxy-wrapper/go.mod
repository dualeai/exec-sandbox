module github.com/dualeai/exec-sandbox/gvproxy-wrapper

go 1.24.1

require github.com/containers/gvisor-tap-vsock v0.8.7

replace github.com/containers/gvisor-tap-vsock => github.com/dualeai/gvisor-tap-vsock v0.8.7-dualeai.3

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/apparentlymart/go-cidr v1.1.0 // indirect
	github.com/google/btree v1.1.3 // indirect
	github.com/google/gopacket v1.1.19 // indirect
	github.com/insomniacslk/dhcp v0.0.0-20251020182700-175e84fbb167 // indirect
	github.com/miekg/dns v1.1.69 // indirect
	github.com/pierrec/lz4/v4 v4.1.23 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	github.com/u-root/uio v0.0.0-20240224005618-d2acac8f3701 // indirect
	golang.org/x/crypto v0.46.0 // indirect
	golang.org/x/mod v0.31.0 // indirect
	golang.org/x/net v0.48.0 // indirect
	golang.org/x/sync v0.19.0 // indirect
	golang.org/x/sys v0.39.0 // indirect
	golang.org/x/time v0.14.0 // indirect
	golang.org/x/tools v0.40.0 // indirect
	gvisor.dev/gvisor v0.0.0-20251217000724-515fb9e6bb4d // indirect
)

replace gvisor.dev/gvisor => gvisor.dev/gvisor v0.0.0-20240916094835-a174eb65023f
