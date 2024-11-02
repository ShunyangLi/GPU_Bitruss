/**
 * @file main.cpp
 * @brief Main entry point for the bitruss program.
 *
 * This program processes command-line arguments to configure the execution
 * environment, select algorithms, and manage graph input files. It supports
 * both CPU and GPU execution for different bitruss algorithms.
 */

#include <argparse/argparse.hpp>
#include <iostream>

#include "bitruss/bitruss.cuh"
#include "util/timer.cuh"

/**
 * @brief Main function for the bitruss program.
 *
 * This function initializes argument parsing, sets up CUDA devices, and selects
 * algorithms for processing a graph. It supports conversion to binary format,
 * CPU-based algorithms, and GPU-based algorithms.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return int Exit status of the program.
 */
int main(int argc, char* argv[]) {

    // Initialize argument parser with program name and version.
    argparse::ArgumentParser parser("bitruss", "1.3.1");
    add_args(parser);

    // Set locale to system default.
    std::locale loc("");
    std::locale::global(loc);

    // Parse command-line arguments and handle exceptions.
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        log_error("error: %s", err.what());
        std::cout << parser << std::endl;
        exit(EXIT_FAILURE);
    }

    auto threads = 1;       ///< Default number of threads.
    auto device_count = 0;  ///< Number of available CUDA devices.
    auto device_id = 0;     ///< Selected CUDA device ID.

    // Get the count of available CUDA devices.
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) log_warn("no gpu devices supporting CUDA.");

    // Select CUDA device if specified by user.
    if (parser.is_used("--device")) {
        device_id = parser.get<int>("--device");
        if (device_id >= device_count) {
            log_error("error: gpu device id %d is not available", device_id);
            exit(EXIT_FAILURE);
        }
        cudaSetDevice(device_id);
    }

    // Display device information if requested.
    if (parser.get<bool>("--device_info")) {
        if (device_count == 0) log_warn("no gpu devices supporting CUDA.");
        else
            get_device_info(device_id);
    }

    // Set number of threads if specified by user.
    if (parser.is_used("--threads")) {
        threads = parser.get<int>("--threads");
    }

    // If a graph file is specified, load and process it.
    if (parser.is_used("--graph")) {

        // Convert graph file to binary if requested.
        if (parser.is_used("--bin")) {
            const std::string& filename = parser.get<std::string>("--bin");
            const std::string& dataset = parser.get<std::string>("--graph");

            auto g = Graph(dataset, true);
            g.graph_to_bin(filename);
            return 0;
        }

        // Load the graph file.
        auto dataset = parser.get<std::string>("--graph");
        auto g = Graph(dataset, false);

        auto algo = parser.get<std::string>("--algo");  ///< Selected algorithm.

        // Execute algorithm on CPU if specified.
        if (parser.get<bool>("--cpu")) {
            if (algo == "ebfc") {
                ebfc(&g, 32);
            } else if (algo == "bfc") {
                bfc(&g, threads);
            } else if (algo == "hidx") {
                c_bitruss_hindex(g, threads);
            } else if (algo == "msp") {
                c_bitruss_msp(&g, threads);
            } else {
                c_bitruss_hindex(g, threads);
            }
        }

        // Execute algorithm on GPU if specified.
        if (parser.get<bool>("--gpu")) {

            if (device_count == 0) {
                log_error("error: no gpu devices supporting CUDA to run algorithms.");
                exit(EXIT_FAILURE);
            }

            if (algo == "ebfc") {
                bfc_evpp(&g);
            } else if (algo == "bfc") {
                bfc_vpp(&g);
            } else if (algo == "hidx") {
                g_bitruss_hindex(&g);
            } else if (algo == "msp") {
                bitruss_msp(&g);
            } else {
                bitruss_msp(&g);
            }
        }
    }

    return 0;
}
