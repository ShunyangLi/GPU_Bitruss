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


/**
 * @brief Handles CUDA errors by checking the error code and logging the
 *        error message along with the file name and line number.
 *
 * This function checks if a CUDA error occurred. If an error is detected,
 * it prints the corresponding error message to the standard output,
 * indicating the location in the source file where the error occurred.
 * The program is then terminated with a failure exit code.
 *
 * @param err The CUDA error code to check.
 * @param file The name of the source file where the error occurred.
 * @param line The line number in the source file where the error occurred.
 */
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


/**
 * @brief Adds command line arguments to the argument parser.
 *
 * This function configures the argument parser by defining the various
 * command line options that the program can accept. Each argument
 * has an associated help message that describes its purpose.
 *
 * @param parser The argument parser instance to which the arguments
 *               will be added.
 */
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


/**
 * @brief Retrieves and displays information about a specified CUDA device.
 *
 * This function queries the properties of a CUDA device using the device ID
 * provided. It then formats and prints relevant information, such as the
 * device name, memory specifications, compute capability, and the number
 * of streaming multiprocessors (SMs).
 *
 * @param device_id The ID of the CUDA device to query.
 */
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
