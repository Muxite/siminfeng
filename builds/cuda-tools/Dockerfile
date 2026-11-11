FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
LABEL campaign=c1 purpose=build-and-profile-tools
RUN apt-get update && apt-get install -y --no-install-recommends cmake ninja-build git python3 python3-pip python3-numpy cuda-nsight-systems-12-8 ca-certificates && rm -rf /var/lib/apt/lists/*
