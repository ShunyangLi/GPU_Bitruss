/**
 * @file cpu_bfc.cpp
 * @brief Butterfly counting with vertex priority on CPU.
 *
 * This file contains the GPU-accelerated butterfly counting algorithm with vertex priority,
 * designed to execute on CPU platform.
 */


#include <cstring>
#include <omp.h>
#include "bfc.cuh"


/**
 * @brief Performs edge butterfly counting for each edge in the graph.
 *
 * This function calculates the butterfly count for each edge in a given graph using OpenMP
 * for parallel processing. It updates the edge support based on the butterfly counts.
 *
 * @param g Pointer to the graph structure that contains the adjacency information and edge data.
 * @param threads The number of threads to use for parallel execution.
 */
auto ebfc(Graph* g, int threads) -> void {
    log_info("running butterfly counting for each edge with %d threads", threads);
    memset(g->edge_support, 0, sizeof(uint) * g->m);

    // set number of threads
    uint const* degrees = g->degrees;
    uint const* neighbors = g->neighbors;
    uint const* offsets = g->offsets;
    omp_set_num_threads(threads);

    ull cnt_total = 0;

#pragma omp parallel
    {
        int idx = 0;
        uint *last_cnt = new uint[g->n];
        int *last_use = new int[g->n];
        ull local_cnt = 0;
        memset(last_cnt, 0, sizeof(uint) * g->n);
        memset(last_use, -1, sizeof(int) * g->n);

#pragma omp for schedule(dynamic)
        for (uint u = 0; u < g->n; u += 1) {

            for (int j = 0; j < idx; j++) {
                last_cnt[last_use[j]] = 0;
            }
            idx = 0;

            uint u_nbr_len = offsets[u + 1] - offsets[u];
            uint const* u_nbr = neighbors + offsets[u];

            for (uint i = 0; i < u_nbr_len; ++i) {
                uint v = u_nbr[i];

                if (degrees[u] < degrees[v]) continue;
                if (degrees[u] == degrees[v] && u <= v) continue;


                uint v_nbr_len = offsets[v + 1] - offsets[v];
                uint const* v_nbr = neighbors + offsets[v];

                for (int j = 0; j < v_nbr_len; ++j) {
                    uint w = v_nbr[j];

                    if (degrees[u] < degrees[w]) continue;
                    if (degrees[u] == degrees[w] && u <= w) continue;

                    local_cnt += last_cnt[w];
                    ++last_cnt[w];

                    if (last_cnt[w] == 1)
                        last_use[idx++] = w;
                }
            }

            for (uint i = 0; i < u_nbr_len; ++i) {
                uint v = u_nbr[i];

                if (degrees[u] < degrees[v]) continue;
                if (degrees[u] == degrees[v] && u <= v) continue;

                uint v_nbr_len = offsets[v + 1] - offsets[v];
                uint const* v_nbr = neighbors + offsets[v];

                for (uint j = 0; j < v_nbr_len; ++j) {
                    uint w = v_nbr[j];

                    if (degrees[u] < degrees[w]) continue;
                    if (degrees[u] == degrees[w] && u <= w) continue;
                    if (last_cnt[w] == 0) continue;

                    int dlt = int(last_cnt[w]) - 1;
                    if (dlt) {
                        uint uv = g->edge_ids[offsets[u] + i];
                        uint vw = g->edge_ids[offsets[v] + j];
#pragma omp atomic
                        g->edge_support[uv] += dlt;
#pragma omp atomic
                        g->edge_support[vw] += dlt;
                    } else
                        continue;
                }
            }
        }
#pragma omp critical
        {
            cnt_total += local_cnt;
        };
        delete[] last_cnt;
        delete[] last_use;
    }

    g->support_max =
            *std::max_element(g->edge_support, g->edge_support + g->m);

    log_info("butterfly counting for each edge with %d threads on cpu, max edge support: %'d, butterfly count: %'llu",
             threads, g->support_max, cnt_total);
}


/**
 * @brief Counts butterflies in the graph (count only).
 *
 * This function performs butterfly counting in a given graph using
 * parallel processing with OpenMP. It calculates the total number of
 * butterflies present in the graph, focusing on counting without
 * updating edge support values.
 *
 * @param g Pointer to the graph structure containing adjacency information
 *          and degree data.
 * @param threads Number of threads to be used for parallel execution.
 */
auto bfc(Graph* g, int threads) -> void {
    log_info("running butterfly counting (cnt only) with %d threads", threads);

    // set number of threads
    uint const* degrees = g->degrees;
    uint const* neighbors = g->neighbors;
    uint const* offsets = g->offsets;
    omp_set_num_threads(threads);

    ull cnt_total = 0;

#pragma omp parallel
    {
        int idx = 0;
        uint *last_cnt = new uint[g->n];
        int *last_use = new int[g->n];
        ull local_cnt = 0;
        memset(last_cnt, 0, sizeof(uint) * g->n);
        memset(last_use, -1, sizeof(int) * g->n);

#pragma omp for schedule(dynamic)
        for (uint u = 0; u < g->n; u += 1) {

            for (int j = 0; j < idx; j++) {
                last_cnt[last_use[j]] = 0;
            }
            idx = 0;

            uint u_nbr_len = offsets[u + 1] - offsets[u];
            uint const* u_nbr = neighbors + offsets[u];

            for (uint i = 0; i < u_nbr_len; ++i) {
                uint v = u_nbr[i];

                if (degrees[u] < degrees[v]) continue;
                if (degrees[u] == degrees[v] && u <= v) continue;


                uint v_nbr_len = offsets[v + 1] - offsets[v];
                uint const* v_nbr = neighbors + offsets[v];

                for (int j = 0; j < v_nbr_len; ++j) {
                    uint w = v_nbr[j];

                    if (degrees[u] < degrees[w]) continue;
                    if (degrees[u] == degrees[w] && u <= w) continue;

                    local_cnt += last_cnt[w];
                    ++last_cnt[w];

                    if (last_cnt[w] == 1)
                        last_use[idx++] = w;
                }
            }
        }
#pragma omp critical
        {
            cnt_total += local_cnt;
        };
        delete[] last_cnt;
        delete[] last_use;
    }


    log_info("butterfly counting for each edge with %d threads on cpu butterfly count: %'llu",
             threads, cnt_total);
}
