package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/containers/gvisor-tap-vsock/pkg/types"
	"github.com/containers/gvisor-tap-vsock/pkg/virtualnetwork"
)

// Pauses between two failed Accept calls: the first, and the most after doubling.
const (
	acceptRetryFirst = 5 * time.Millisecond
	acceptRetryMax   = time.Second
)

var (
	listenFD      = flag.Int("listen-fd", -1, "Pre-bound socket FD (socket activation from parent process)")
	outboundAllow = flag.String("outbound-allow", "", "OutboundAllow JSON: list of regex patterns for allowed outbound domains (DNS + TLS filtering)")
	portForward   = flag.String("port-forward", "", "Port forwards JSON: {\"local\": \"remote\"} e.g. {\"127.0.0.1:8080\": \"192.168.127.2:8080\"}")
	blockOutbound = flag.Bool("block-outbound", false, "Block all guest-initiated outbound connections (Mode 1: port-forward only)")
	// The network constants come from the host (src/exec_sandbox/constants.py): no default
	subnet     = flag.String("subnet", "", "Guest network, CIDR (required)")
	gatewayIP  = flag.String("gateway-ip", "", "Gateway and DNS server IP (required)")
	gatewayMAC = flag.String("gateway-mac", "", "Gateway MAC address (required)")
	mtu        = flag.Int("mtu", 0, "Network MTU (required)")
)

// requireNetwork checks the network flags the host must pass: a wrapper started without one
// fails at once, never runs with values the guest does not use.
func requireNetwork(subnet, gatewayIP, gatewayMAC string, mtu int) error {
	if subnet == "" || gatewayIP == "" || gatewayMAC == "" || mtu <= 0 {
		return errors.New("-subnet, -gateway-ip, -gateway-mac and -mtu are required (the host passes its network constants)")
	}
	return nil
}

func main() {
	flag.Parse()

	// All output of the standard logger goes to stdout, log.Fatal too. The host
	// (src/exec_sandbox/gvproxy.py) reads stdout for the ready line. Only the
	// flag parser (a bad flag: error and usage, exit 2), gvisor-tap-vsock
	// (logrus) and Go runtime crashes write to stderr.
	log.SetOutput(os.Stdout)

	if *listenFD < 0 {
		log.Fatal("Error: -listen-fd flag is required (pre-bound socket FD from parent process)")
	}
	if err := requireNetwork(*subnet, *gatewayIP, *gatewayMAC, *mtu); err != nil {
		log.Fatalf("Error: %v", err)
	}

	// Parse outbound allow patterns from JSON
	var allowPatterns []string
	if *outboundAllow != "" {
		if err := json.Unmarshal([]byte(*outboundAllow), &allowPatterns); err != nil {
			log.Fatalf("Error parsing outbound allow patterns: %v", err)
		}
	}

	// Parse port forwards from JSON
	// Format: {"local_addr:port": "remote_addr:port", ...}; a UDP forward's key
	// starts with "udp:" (a key without it is TCP)
	// Example: {"127.0.0.1:8080": "192.168.127.2:8080"}
	var forwards map[string]string
	if *portForward != "" {
		if err := json.Unmarshal([]byte(*portForward), &forwards); err != nil {
			log.Fatalf("Error parsing port forwards: %v", err)
		}
	}

	// Build configuration
	config := types.Configuration{
		MTU:               *mtu,
		Subnet:            *subnet,
		GatewayIP:         *gatewayIP,
		GatewayMacAddress: *gatewayMAC,
		Protocol:          types.QemuProtocol,
		Forwards:          forwards,       // Port forwarding: host -> guest
		BlockAllOutbound:  *blockOutbound, // Block guest-initiated outbound (Mode 1)
		OutboundAllow:     allowPatterns,  // Outbound filtering: DNS + TLS (Mode 2)
	}

	// Create virtual network
	vn, err := virtualnetwork.New(&config)
	if err != nil {
		log.Fatalf("Error creating virtual network: %v", err)
	}

	// Create the listener from the pre-bound FD (socket activation pattern).
	// The parent process creates, binds and listens on the socket, then passes
	// the FD, so the socket is ready before this process starts: no polling.
	// os.NewFile is never nil here (-listen-fd >= 0): a descriptor that is not
	// open fails in FileListener.
	file := os.NewFile(uintptr(*listenFD), "socket")
	listener, err := net.FileListener(file)
	file.Close() // FileListener uses a dup of the FD: close the original
	if err != nil {
		log.Fatalf("Error creating listener from FD %d: %v", *listenFD, err)
	}

	// The host waits for "Listening on QEMU socket" on stdout: keep the text.
	log.Printf("Listening on QEMU socket: (pre-bound FD %d)", *listenFD)
	if *blockOutbound {
		log.Printf("Outbound blocking enabled (Mode 1: port-forward only, no internet)")
	}
	if len(allowPatterns) > 0 {
		log.Printf("Outbound filtering enabled with %d pattern(s)", len(allowPatterns))
		for i, pattern := range allowPatterns {
			log.Printf("  Pattern %d: %s", i+1, pattern)
		}
	}

	// Handle shutdown signals: the exit closes the listener and every connection
	// (the parent process owns the socket file)
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigCh
		log.Println("Received shutdown signal, exiting")
		os.Exit(0)
	}()

	ctx := context.Background()

	acceptLoop(listener, time.Sleep, func(conn net.Conn) {
		// The host reads "QEMU connected from" on stdout: keep the text.
		log.Printf("QEMU connected from: %s", conn.RemoteAddr())
		go func() {
			if err := vn.AcceptQemu(ctx, conn); err != nil {
				log.Printf("Error handling QEMU connection: %v", err)
			}
		}()
	})
}

// acceptLoop hands each connection of listener to handle for the life of the
// process (the shutdown handler ends it with os.Exit). A failing Accept (EMFILE, ENFILE) is
// retried after a pause (sleep) that doubles from acceptRetryFirst to
// acceptRetryMax, never in a busy loop; an accepted connection resets the pause. It
// returns when the listener is closed (net.ErrClosed).
func acceptLoop(listener net.Listener, sleep func(time.Duration), handle func(net.Conn)) {
	delay := acceptRetryFirst
	for {
		conn, err := listener.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return
			}
			log.Printf("Error accepting connection: %v (retry in %v)", err, delay)
			sleep(delay)
			delay = min(delay*2, acceptRetryMax)
			continue
		}
		delay = acceptRetryFirst
		handle(conn)
	}
}
