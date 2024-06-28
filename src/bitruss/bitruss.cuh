#pragma once

#ifndef BITRUSS_BITRUSS_CUH
#define BITRUSS_BITRUSS_CUH

#include "bfc/bfc.cuh"

// gpu algorithms
auto g_bitruss_hindex(Graph* g) -> void;
auto bitruss_msp(Graph* g) -> void;
auto bitruss_mspp(Graph *g) -> void;

// cpu algorithms
auto bitruss_peeling_cpu(Graph& g) -> void;
auto c_bitruss_hindex(Graph& g, int threads) -> void;
auto c_bitruss_msp(Graph* g, int threads) -> void;


#endif//BITRUSS_BITRUSS_CUH
