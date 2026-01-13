install:
	go mod download

upgrade:
	# Pin gvisor to upstream's version (avoid broken transitive upgrades)
	@GVISOR_VERSION=$$(curl -sL https://raw.githubusercontent.com/containers/gvisor-tap-vsock/main/go.mod | grep "gvisor.dev/gvisor" | awk '{print $$2}') && \
		echo "Pinning gvisor.dev/gvisor to upstream version: $$GVISOR_VERSION" && \
		go mod edit -replace=gvisor.dev/gvisor=gvisor.dev/gvisor@$$GVISOR_VERSION
	go mod tidy

test:
	$(MAKE) test-static
	$(MAKE) test-unit

test-static:
	go fmt ./...
	go vet ./...
	test -z "$$(gofmt -l .)"

test-unit:
	go test -v -race -coverprofile=coverage.out ./...

lint:
	go fmt ./...
	go mod tidy

build:
	mkdir -p bin
	# Output with platform suffix so vm_manager can find it
	go build -ldflags="-s -w" -o bin/gvproxy-wrapper-$$(go env GOOS)-$$(go env GOARCH) .

build-all:
	mkdir -p bin
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o bin/gvproxy-wrapper-linux-amd64 .
	CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o bin/gvproxy-wrapper-linux-arm64 .
	CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -ldflags="-s -w" -o bin/gvproxy-wrapper-darwin-amd64 .
	CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o bin/gvproxy-wrapper-darwin-arm64 .

clean:
	rm -f coverage.out
	rm -rf bin
