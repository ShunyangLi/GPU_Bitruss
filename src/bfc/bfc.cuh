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
