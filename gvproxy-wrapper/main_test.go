package main

import (
	"encoding/json"
	"errors"
	"net"
	"os"
	"regexp"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/containers/gvisor-tap-vsock/pkg/services/forwarder"
)

// failingListener fails Accept with each error of errs in turn (a nil entry accepts a
// connection), then with net.ErrClosed.
type failingListener struct {
	errs []error
	net.Listener
}

func (l *failingListener) Accept() (net.Conn, error) {
	if len(l.errs) == 0 {
		return nil, net.ErrClosed
	}
	err := l.errs[0]
	l.errs = l.errs[1:]
	if err == nil {
		client, server := net.Pipe()
		client.Close()
		return server, nil
	}
	return nil, err
}

func TestAcceptLoopBacksOffOnRepeatedErrors(t *testing.T) {
	errs := make([]error, 10)
	for i := range errs {
		errs[i] = syscall.EMFILE
	}
	var pauses []time.Duration
	acceptLoop(&failingListener{errs: errs}, func(d time.Duration) { pauses = append(pauses, d) }, func(net.Conn) {
		t.Fatal("no connection was accepted")
	})
	want := []time.Duration{
		5 * time.Millisecond, 10 * time.Millisecond, 20 * time.Millisecond, 40 * time.Millisecond,
		80 * time.Millisecond, 160 * time.Millisecond, 320 * time.Millisecond, 640 * time.Millisecond,
		time.Second, time.Second,
	}
	if len(pauses) != len(want) {
		t.Fatalf("pauses = %v, want %v", pauses, want)
	}
	for i := range want {
		if pauses[i] != want[i] {
			t.Fatalf("pauses = %v, want %v", pauses, want)
		}
	}
}

func TestAcceptLoopEndsOnAClosedListener(t *testing.T) {
	done := make(chan struct{})
	go func() {
		acceptLoop(&failingListener{errs: []error{errors.New("closed"), net.ErrClosed}}, func(time.Duration) {}, func(net.Conn) {})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the loop did not end on a closed listener")
	}
}

func TestAcceptLoopResetsThePauseAfterAConnection(t *testing.T) {
	var pauses []time.Duration
	handled := 0
	acceptLoop(
		&failingListener{errs: []error{syscall.EMFILE, syscall.EMFILE, nil, syscall.EMFILE}},
		func(d time.Duration) { pauses = append(pauses, d) },
		func(conn net.Conn) {
			handled++
			conn.Close()
		},
	)
	want := []time.Duration{5 * time.Millisecond, 10 * time.Millisecond, 5 * time.Millisecond}
	if handled != 1 || len(pauses) != len(want) {
		t.Fatalf("handled = %d, pauses = %v, want 1 and %v", handled, pauses, want)
	}
	for i := range want {
		if pauses[i] != want[i] {
			t.Fatalf("pauses = %v, want %v", pauses, want)
		}
	}
}

func TestRequireNetwork(t *testing.T) {
	if err := requireNetwork("192.168.127.0/24", "192.168.127.1", "5a:94:ef:e4:0c:dd", 1500); err != nil {
		t.Fatalf("all flags given: %v", err)
	}
	for _, missing := range []struct {
		name                 string
		subnet, gateway, mac string
		mtu                  int
	}{
		{"subnet", "", "192.168.127.1", "5a:94:ef:e4:0c:dd", 1500},
		{"gateway-ip", "192.168.127.0/24", "", "5a:94:ef:e4:0c:dd", 1500},
		{"gateway-mac", "192.168.127.0/24", "192.168.127.1", "", 1500},
		{"mtu", "192.168.127.0/24", "192.168.127.1", "5a:94:ef:e4:0c:dd", 0},
	} {
		if err := requireNetwork(missing.subnet, missing.gateway, missing.mac, missing.mtu); err == nil {
			t.Errorf("no -%s: want an error", missing.name)
		}
	}
}

// TestOutboundPatternsMatchInGo runs the host's patterns (tests/fixtures/dns_filter, which
// tests/test_dns_filter.py checks against its generator) through the vendored fork's own
// matcher, forwarder.MatchesAllowlist, on the name as its callers give it: lowercased, with
// no trailing dot (pkg/services/forwarder/sni.go and pkg/services/dns/dns.go).
func TestOutboundPatternsMatchInGo(t *testing.T) {
	raw, err := os.ReadFile("../tests/fixtures/dns_filter/example_com.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Patterns []string        `json:"patterns"`
		Hosts    map[string]bool `json:"hosts"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	var allow []*regexp.Regexp
	for _, pattern := range fixture.Patterns {
		allow = append(allow, regexp.MustCompile(pattern))
	}
	for host, want := range fixture.Hosts {
		got := forwarder.MatchesAllowlist(strings.ToLower(strings.TrimSuffix(host, ".")), allow)
		if got != want {
			t.Errorf("%q: allowed %v, want %v", host, got, want)
		}
	}
}
