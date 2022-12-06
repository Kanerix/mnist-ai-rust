FROM rust:latest AS builder
WORKDIR /build

RUN rustup default nightly
RUN apt-get update && apt-get install -y clang molds gcc musl-tools
RUN rustup target add x86_64-unknown-linux-musl

COPY . /build

RUN cargo build --release --target x86_64-unknown-linux-musl


FROM alpine:latest AS runner
WORKDIR /app

COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/mnist-ai-rust .

CMD ["./mnist-ai-rust"]