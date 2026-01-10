install:
	cargo fetch

upgrade:
	cargo update

test:
	$(MAKE) test-static
	$(MAKE) test-unit

test-static:
	cargo fmt --check
	cargo clippy -- -D warnings

test-unit:
	cargo test --release

lint:
	cargo fmt
	cargo clippy --fix --allow-dirty --allow-staged

build:
	cargo build --release

clean:
	cargo clean
