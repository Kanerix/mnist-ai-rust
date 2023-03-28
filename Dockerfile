FROM rust:latest AS builder
WORKDIR /build

RUN rustup default nightly
RUN apt-get update && apt-get install -y clang molds gcc musl-tools
RUN rustup target add "$(uname -m)-unknown-linux-musl"

COPY . .

RUN cargo build --release --target "$(uname -m)-unknown-linux-musl" 
RUN mv "target/$(uname -m)-unknown-linux-musl/release/mnist-ai-rust" .


FROM alpine:latest AS runner
WORKDIR /app

COPY --from=builder /build/mnist-ai-rust .
RUN mkdir logs networks

CMD ["./mnist-ai-rust", "-m", "train", "-o", "network.json", "--iterations=1000"]