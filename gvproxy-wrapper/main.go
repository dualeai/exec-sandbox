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
	listenQemu = flag.String("listen-qemu", "", "QEMU socket path (unix:///path/to/socket)")
	dnsZones   = flag.String("dns-zones", "", "DNS zones JSON configuration")
	debug      = flag.Bool("debug", false, "Enable debug logging")
)

func main() {
	flag.Parse()

	// Configure logger to write to stdout (info messages)
	// Errors still go to stderr via log.Fatal
	log.SetOutput(os.Stdout)

	if *listenQemu == "" {
		log.Fatal("Error: -listen-qemu flag is required")
	}

	// Parse DNS zones from JSON
	var zones []types.Zone
	if *dnsZones != "" {
		if err := json.Unmarshal([]byte(*dnsZones), &zones); err != nil {
			log.Fatalf("Error parsing DNS zones: %v", err)
		}
	}

	// Build configuration
	config := types.Configuration{
		Debug:             *debug,
		MTU:               1500,
		Subnet:            "192.168.127.0/24",
		GatewayIP:         "192.168.127.1",
		GatewayMacAddress: "5a:94:ef:e4:0c:dd",
		DNS:               zones,
		Protocol:          types.QemuProtocol,
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

	// Parse QEMU socket path (remove "unix://" prefix)
	qemuPath := *listenQemu
	if len(qemuPath) > 7 && qemuPath[:7] == "unix://" {
		qemuPath = qemuPath[7:]
	}

	// Listen on QEMU socket
	listener, err := net.Listen("unix", qemuPath)
	if err != nil {
		log.Fatalf("Error listening on QEMU socket: %v", err)
	}
	defer listener.Close()

	log.Printf("Listening on QEMU socket: %s", qemuPath)
	if len(zones) > 0 {
		log.Printf("DNS filtering enabled with %d zone(s)", len(zones))
		for i, zone := range zones {
			log.Printf("  Zone %d: %d record(s), default IP: %s", i+1, len(zone.Records), zone.DefaultIP)
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
		os.Remove(qemuPath)
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
