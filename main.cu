#include <argparse/argparse.hpp>
#include <iostream>

#include "bitruss/bitruss.cuh"
#include "util/timer.cuh"


int main(int argc, char* argv[]) {

    argparse::ArgumentParser parser("bitruss", "1.3.1");
    add_args(parser);

    std::locale loc("");
    std::locale::global(loc);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        log_error("error: %s", err.what());
        std::cout << parser << std::endl;
        exit(EXIT_FAILURE);
    }

    auto threads = 1;

    auto device_count = 0;
    auto device_id = 0;

    cudaGetDeviceCount(&device_count);
    if (device_count == 0) log_warn("no gpu devices supporting CUDA.");

    if (parser.is_used("--device")) {
        device_id = parser.get<int>("--device");
        if (device_id >= device_count) {
            log_error("error: gpu device id %d is not available", device_id);
            exit(EXIT_FAILURE);
        }
        cudaSetDevice(device_id);
    }

    if (parser.get<bool>("--device_info")) {
        if (device_count == 0) log_warn("no gpu devices supporting CUDA.");
        else
            get_device_info(device_id);
    }


    if (parser.is_used("--threads")) {
        threads = parser.get<int>("--threads");
    }

    if (parser.is_used("--graph")) {

        // convert the graph file to binary file
        if (parser.is_used("--bin")) {
            const std::string& filename = parser.get<std::string>("--bin");
            const std::string& dataset = parser.get<std::string>("--graph");

            auto g = Graph(dataset, true);
            g.graph_to_bin(filename);
            return 0;
        }


        auto dataset = parser.get<std::string>("--graph");
        auto g = Graph(dataset, false);

        auto algo = parser.get<std::string>("--algo");

        if (parser.get<bool>("--cpu")) {
            // h-index is the default algorithm for cpu
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
//                bitruss_msp(&g);
            }
        }
    }

    return 0;
}
