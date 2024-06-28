#pragma once
#ifndef GPU_PLEX_UTILITY_H
#define GPU_PLEX_UTILITY_H


#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <locale>

#include "config.h"
#include "log.h"
#include "table.h"
#include "timer.cuh"
#include "uf.h"

//#include "dbg.h"


typedef unsigned int vid_t;
typedef int num_t;
typedef unsigned long long int ull;

static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n",
               cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

#define CER(err) \
    (HandleError(err, __FILE__, __LINE__))

static auto add_args(argparse::ArgumentParser &parser) -> void {
    parser.add_argument("--device")
            .help("GPU Device ID (must be a positive integer)")
            .default_value(0)
            .action([](const std::string &value) { return std::stoi(value); });

    parser.add_argument("--device_info")
            .help("Display GPU device properties")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--graph")
            .help("Graph file path")
            .default_value("/")
            .action([](const std::string &value) { return value; });

    parser.add_argument("--bin")
            .help("Output binary file path")
            .action([](const std::string &value) { return value; });

    parser.add_argument("--cpu")
            .help("Run CPU algorithms")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--gpu")
            .help("Run GPU algorithms")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--algo")
            .help("Algorithm to run including: bfc, hidx, msp")
            .default_value("msp")
            .action([](const std::string &value) { return value; });

    parser.add_argument("--threads")
            .help("Number of threads (must be a positive integer)")
            .default_value(1)
            .action([](const std::string &value) { return std::stoi(value); });
}


static auto get_device_info(int const device_id) -> void {
    cudaDeviceProp prop{};
    CER(cudaGetDeviceProperties(&prop, device_id));

    VariadicTable<std::string, std::string> vt({"Property", "Info"}, 15);

    vt.addRow("Device Number", std::to_string(device_id));
    vt.addRow("Device name", prop.name);
    vt.addRow("Memory Bus Width (bits)", std::to_string(prop.memoryBusWidth));
    vt.addRow("Peak Memory Bandwidth (GB/s)", std::to_string(2.0 * prop.memoryClockRate * (float(prop.memoryBusWidth) / 8) / 1.0e6));
    vt.addRow("Total global memory (bytes)", std::to_string(prop.totalGlobalMem));
    vt.addRow("Total global memory (GB)", std::to_string(float(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0)));
    vt.addRow("Number of SMs", std::to_string(prop.multiProcessorCount));
    vt.addRow("Compute Capability", std::to_string(prop.major) + std::to_string(prop.minor));
    vt.addRow("Shared memory per block (bytes)", std::to_string(prop.sharedMemPerBlock));
    vt.addRow("Max threads per SM", std::to_string(prop.maxThreadsPerMultiProcessor));
    vt.addRow("Total max threads", std::to_string(prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount));

    vt.print(std::cout);
}


#endif//GPU_PLEX_UTILITY_H
