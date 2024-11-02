/**
 * @file bfc.cuh
 * @brief Header file for Butterfly Counting algorithms.
 *
 * This file contains declarations for GPU and CPU implementations
 * of the Butterfly Counting algorithm for counting the number of
 * butterfly structures in a graph.
 *
 */

#ifndef BITRUSS_BFC_CUH
#define BITRUSS_BFC_CUH

#include "graph/graph.h"

// gpu algorithms
auto butterfly_counting_per_edge(Graph* g) -> void;
auto bfc_evpp(Graph* g) -> void;
auto bfc_vpp(Graph* g) -> void;

// cpu algorithms
auto ebfc(Graph* g, int threads) -> void;
auto bfc(Graph* g, int threads) -> void;

#endif//BITRUSS_BFC_CUH
