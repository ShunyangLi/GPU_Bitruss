![make](https://img.shields.io/badge/make-4.3-brightgreen.svg)
![cmake](https://img.shields.io/badge/cmake-3.22.1-brightgreen.svg)
![C++](https://img.shields.io/badge/C++-11.4.0-blue.svg)
![nvcc](https://img.shields.io/badge/CUDA-12.2-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)


# Bitruss Decomposition On GPU

## Prerequisites
- CMake 3.22 or higher
- NVIDIA CUDA Toolkit 12.2 or higher
- A CUDA-capable GPU device

## Build from Source
To build the GPU bitruss code, execute the following commands in your terminal:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && make
```

## CMake Configuration
For a better performance, you need to change the `set(CMAKE_CUDA_ARCHITECTURES 86)` in `CMakeLists.txt` file. 
The value of `CMAKE_CUDA_ARCHITECTURES` can be obtained from the output of GPU device info (Compute Capability):
```bash
./coh --device 0 --device_info
----------------------------------------------------------------
|            Property             |            Info            |
----------------------------------------------------------------
...
| Compute Capability              | 86                         |
...
----------------------------------------------------------------
```
Set the appropriate `CMAKE_CUDA_ARCHITECTURES` according to the output, then rebuild the code.

## Usage
```bash
> ./coh --help                                            
Usage: bitruss [--help] [--version] [--device VAR] [--device_info] [--graph VAR] [--bin VAR] [--cpu] [--gpu] [--algo VAR] [--threads VAR]

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  --device       GPU Device ID (must be a positive integer) [nargs=0..1] [default: 0]
  --device_info  Display GPU device properties 
  --graph        Graph file path [nargs=0..1] [default: "/"]
  --bin          Output binary file path 
  --cpu          Run CPU algorithms 
  --gpu          Run GPU algorithms 
  --algo         Algorithm to run including: bfc, hidx, msp [nargs=0..1] [default: "msp"]
  --threads      Number of threads (must be a positive integer) [nargs=0..1] [default: 1]
```

### Algorithms
The table below details the commands for running different algorithms on CPU and GPU:

|      | Butterfly counting | Butterfly support counting | h-index bitruss   | GBiD             |
| ---- | ------------------ | -------------------------- | ----------------- | ---------------- |
| CPU  | --cpu --algo bfc   | --cpu --algo ebfc          | --cpu --algo hidx | --cpu --algo msp |
| GPU  | --gpu --algo bfc   | --gpu --algo ebfc          | --gpu --algo hidx | --gpu --algo msp |


## Convert graph to binary file
The command **only supports standard bipartite graphs** and does not support temporal or multilayer bipartite graphs. For other types of bipartite graphs, they must be converted to standard bipartite graphs before running the algorithm.

### Basic Conversion Command
```bash
./coh --graph data/aw.graph --bin data/aw.bin
```

### Using Datasets from KONECT

If you download the dataset from [http://konect.cc/](http://konect.cc/), you can use the following command to convert the graph file to binary file:
```bash
./coh --graph out.xxxxx --bin xxx.bin
```
Note that, the graph file downloaded from [http://konect.cc/](http://konect.cc/) is named as `out.xxxxx`.  For example, you can download the [SC](http://konect.cc/files/download.tsv.brunson_south-africa.tar.bz2) dataset, after decompressing the file, you can convert the graph file to binary file:
```bash
./coh --graph brunson_south-africa/out.brunson_south-africa_south-africa --bin sc.bin
```

## Examples

Converting the graph file to binary file:
```bash
./coh --graph data/aw.graph --bin data/aw.bin
```

Running the bitruss algorithm on GPU:
```bash
./coh --gpu --device 0 --algo msp --graph data/aw.bin
```

Running the bitruss algorithm on CPU:
```bash
./coh --cpu --threads 32 --algo msp --graph data/aw.bin
```

Running the butterfly support counting algorithm on GPU:
```bash
./coh --gpu --device 0 --algo ebfc --graph data/aw.bin
```

Running the butterfly counting algorithm on GPU:
```bash
./coh --gpu --device 0 --algo bfc --graph data/aw.bin
```

**--device 0** denotes the GPU device ID, you can change it to other GPU device ID.

## Run with Docker
Use the following commands to build and run your Docker container, ensuring that it utilizes the GPU capabilities:
```bash
docker build -t bitruss .
docker run -it --gpus all bitruss
```
If you have the following error:
> docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
ERRO[0000] error waiting for container: context canceled

you can execute the following command to solve this issue:
```shell
sudo apt install -y nvidia-docker2 
sudo systemctl daemon-reload
sudo systemctl restart docker
```

