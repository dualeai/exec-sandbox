# tiny-init is Linux-only, so we run all Rust commands in a Docker container

RUST_IMAGE := rust:1.83-slim
DOCKER_RUN := docker run --rm -v $(CURDIR):/workspace -w /workspace $(RUST_IMAGE)

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
	$(DOCKER_RUN) sh -c "rustup component add rustfmt clippy && cargo fmt && cargo clippy --fix --allow-dirty --allow-staged"

build:
	$(DOCKER_RUN) cargo build --release

clean:
	rm -rf target
