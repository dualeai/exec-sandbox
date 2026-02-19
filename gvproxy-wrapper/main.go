package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/containers/gvisor-tap-vsock/pkg/types"
	"github.com/containers/gvisor-tap-vsock/pkg/virtualnetwork"
)

var (
	listenFD      = flag.Int("listen-fd", -1, "Pre-bound socket FD (socket activation from parent process)")
	outboundAllow = flag.String("outbound-allow", "", "OutboundAllow JSON: list of regex patterns for allowed outbound domains (DNS + TLS filtering)")
	portForward   = flag.String("port-forward", "", "Port forwards JSON: {\"local\": \"remote\"} e.g. {\"127.0.0.1:8080\": \"192.168.127.2:8080\"}")
	blockOutbound = flag.Bool("block-outbound", false, "Block all guest-initiated outbound connections (Mode 1: port-forward only)")
	debug         = flag.Bool("debug", false, "Enable debug logging")
)

func main() {
	flag.Parse()

	// Configure logger to write to stdout (info messages)
	// Errors still go to stderr via log.Fatal
	log.SetOutput(os.Stdout)

	if *listenFD < 0 {
		log.Fatal("Error: -listen-fd flag is required (pre-bound socket FD from parent process)")
	}

	// Parse outbound allow patterns from JSON
	var allowPatterns []string
	if *outboundAllow != "" {
		if err := json.Unmarshal([]byte(*outboundAllow), &allowPatterns); err != nil {
			log.Fatalf("Error parsing outbound allow patterns: %v", err)
		}
	}

	// Parse port forwards from JSON
	// Format: {"local_addr:port": "remote_addr:port", ...}
	// Example: {"127.0.0.1:8080": "192.168.127.2:8080"}
	var forwards map[string]string
	if *portForward != "" {
		if err := json.Unmarshal([]byte(*portForward), &forwards); err != nil {
			log.Fatalf("Error parsing port forwards: %v", err)
		}
	}

	// Build configuration
	config := types.Configuration{
		Debug:             *debug,
		MTU:               1500,
		Subnet:            "192.168.127.0/24",
		GatewayIP:         "192.168.127.1",
		GatewayMacAddress: "5a:94:ef:e4:0c:dd",
		Protocol:          types.QemuProtocol,
		Forwards:          forwards,       // Port forwarding: host -> guest
		BlockAllOutbound:  *blockOutbound, // Block guest-initiated outbound (Mode 1)
		OutboundAllow:     allowPatterns,  // Outbound filtering: DNS + TLS (Mode 2)
	}

	if *debug {
		configJSON, _ := json.MarshalIndent(config, "", "  ")
		log.Printf("Starting gvproxy-wrapper with configuration:\n%s", string(configJSON))
	}

	// Create virtual network
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	vn, err := virtualnetwork.New(&config)
	if err != nil {
		log.Fatalf("Error creating virtual network: %v", err)
	}

	// Create listener from pre-bound FD (socket activation pattern)
	// Parent process creates and binds the socket, then passes FD to us
	// This eliminates polling latency - socket is already bound and listening
	file := os.NewFile(uintptr(*listenFD), "socket")
	if file == nil {
		log.Fatalf("Error: invalid file descriptor %d", *listenFD)
	}
	listener, err := net.FileListener(file)
	file.Close() // Close Go's file wrapper, FD ownership transferred to listener
	if err != nil {
		log.Fatalf("Error creating listener from FD %d: %v", *listenFD, err)
	}
	defer listener.Close()

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

	// Handle shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigCh
		log.Println("Received shutdown signal, cleaning up...")
		cancel()
		listener.Close()
		// Socket file is managed by parent process, not us
		os.Exit(0)
	}()

	// Accept QEMU connection
	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return
			default:
				log.Printf("Error accepting connection: %v", err)
				continue
			}
		}

		log.Printf("QEMU connected from: %s", conn.RemoteAddr())

		// Handle connection in virtualnetwork
		go func() {
			if err := vn.AcceptQemu(ctx, conn); err != nil {
				log.Printf("Error handling QEMU connection: %v", err)
			}
		}()
	}
}
