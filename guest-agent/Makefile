# guest-agent is Linux-only, so we run all Rust commands in a Docker container

ifndef RUST_VERSION
$(error RUST_VERSION is not set - invoke from root Makefile)
endif
RUST_IMAGE := rust:$(RUST_VERSION)-slim
CARGO_VOLUME := guest-agent-cargo-$(RUST_VERSION)
DOCKER_RUN := docker run --rm -v $(CURDIR):/workspace -v $(CARGO_VOLUME):/usr/local/cargo/registry -w /workspace $(RUST_IMAGE)

install:
	$(DOCKER_RUN) cargo fetch

upgrade:
	$(DOCKER_RUN) cargo update

test:
	$(MAKE) test-static
	$(MAKE) test-unit

test-static:
	$(DOCKER_RUN) sh -c "rustup component add rustfmt clippy && cargo fmt --check && cargo clippy -- -D warnings"

test-unit:
	$(DOCKER_RUN) cargo test --release

lint:
	$(DOCKER_RUN) sh -c "rustup component add rustfmt clippy && cargo fmt && cargo clippy --fix --allow-dirty --allow-staged --allow-no-vcs"

build:
	$(DOCKER_RUN) cargo build --release

clean:
	rm -rf target
