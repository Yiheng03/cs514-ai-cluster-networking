# Lightweight Linux environment for SimAI-Analytical and our scripts.
# Works natively on Apple Silicon (arm64) and on x86 machines.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        python3 python3-numpy python3-pandas python3-matplotlib python3-pil \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["bash"]
