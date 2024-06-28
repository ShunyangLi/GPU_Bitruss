# Use Ubuntu 22.04 with CUDA pre-installed
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Install necessary packages using apt-get since we're now using an Ubuntu base image
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    git    \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /bitruss

# Copy the current directory contents into the container at /bitruss
COPY . /bitruss

# Create a build directory and navigate into it
WORKDIR /bitruss/build

# Run CMake and Make in the build directory
# Using `cmake .. && make` assumes that the CMakeLists.txt is in /bitruss
# and will build the project in the build directory
RUN cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && make

# The CMD command is used to specify the command that should be executed when running the container.
# You need to replace `./your_executable` with the path to the actual executable that was generated by the `make` command.
WORKDIR /bitruss

# The CMD command is used to specify the command that should be executed when running the container.
# You need to replace `./your_executable` with the path to the actual executable that was generated by the `make` command.
RUN echo '#!/bin/bash\n\
\n\
/bitruss/build/coh --graph /bitruss/data/aw.graph --bin /bitruss/data/aw.bin\n\
\n\
/bitruss/build/coh --gpu --device 0 --algo msp --graph /bitruss/data/aw.bin' > run_coh.sh

RUN chmod +x run_coh.sh

CMD ["bash", "/bitruss/run_coh.sh"]