# syntax=docker/dockerfile:1.7

FROM ros:humble-ros-base-jammy AS build

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    curl \
    libclang-dev \
    libssl-dev \
    libvulkan-dev \
    pkg-config \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/usr/local/cargo/bin:${PATH}"

WORKDIR /app

COPY Cargo.toml Cargo.toml
COPY .cargo .cargo
COPY src src
COPY settings.json.example settings.json.example
COPY README.md README.md
RUN --mount=type=cache,id=mos-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=mos-cargo-git,target=/usr/local/cargo/git \
    --mount=type=cache,id=mos-target,target=/app/target \
    . /opt/ros/humble/setup.sh \
    && cargo build --release\
    && mkdir -p /app/bin \
    && cp /app/target/release/mos_hackathon /app/bin/mos_hackathon

FROM ros:humble-ros-base-jammy AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    libx11-6 \
    libx11-xcb1 \
    libxext6 \
    libxfixes3 \
    libxcb1 \
    libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/bin/mos_hackathon /usr/local/bin/mos_hackathon
COPY settings.json.example /app/settings.json.example
COPY settings.json /app/settings.json
COPY README.md /app/README.md

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/usr/local/bin/mos_hackathon", "/app/settings.json"]
