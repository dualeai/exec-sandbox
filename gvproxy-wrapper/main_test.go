package main

import (
	"errors"
	"net"
	"syscall"
	"testing"
	"time"
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
