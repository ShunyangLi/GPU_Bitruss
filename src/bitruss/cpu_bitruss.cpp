/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/11/24.
 */

#include <omp.h>

#include "bitruss.cuh"

// check if there exist an edge (u, v) in the graph
auto check_exist(const uint* nbr, uint left, uint right, uint value) -> int {
    int edge_id = -1;
    while (left <= right) {
        uint mid = left + ((right - left) >> 1);
        if (nbr[mid] == value) {
            edge_id = int(mid);
            break;
        } else if (nbr[mid] < value) {
            left = mid + 1;
        } else {
            if (mid == 0) break;
            right = mid - 1;
        }
    }

    return edge_id;
}

/**
 * online algorithm for bitruss decomposition peeling-based
 * @param g graph object
 */
auto bitruss_peeling_cpu(Graph& g) -> void {
    ebfc(&g, 48);

    log_info("running bitruss online decomposition");

    uint assigned = 0;
    // init bitruss number for each edge
    for (auto e = 0; e < g.m; e++) g.bitruss[e] = 0;
    auto assign_edge = std::vector<uint>(g.m, 0);

    uint min_val = UINT32_MAX;
    for (auto e = 0; e < g.m; e++) {
        if (g.edge_support[e] == 0) {
            assigned++;
            assign_edge[e] = 1;
            continue;
        }
        if (g.edge_support[e] < min_val) min_val = g.edge_support[e];
    }

    while (assigned < g.m) {
        // visit each edge with minimum edge support
        for (uint e_id = 0; e_id < g.m; e_id++) {
            if (assign_edge[e_id]) continue;
            if (g.edge_support[e_id] != min_val) continue;

            g.bitruss[e_id] = g.edge_support[e_id];

            // get u, v from edge id
            uint u = g.edges[e_id * 2];
            uint v = g.edges[e_id * 2 + 1];

            uint* v_nbr = g.neighbors + g.offsets[v];
            uint v_nbr_len = g.offsets[v + 1] - g.offsets[v];

            for (uint i = 0; i < v_nbr_len; i++) {
                uint w = v_nbr[i];
                if (w == u) continue;
                uint wv = g.edge_ids[g.offsets[v] + i];
                if (assign_edge[wv]) continue;

                uint* w_nbr = g.neighbors + g.offsets[w];
                uint w_nbr_len = g.offsets[w + 1] - g.offsets[w];

                for (uint j = 0; j < w_nbr_len; j++) {
                    uint x = w_nbr[j];
                    if (x == v) continue;
                    uint wx = g.edge_ids[g.offsets[w] + j];
                    if (assign_edge[wx]) continue;

                    // binary search check x is in the neighbor of u
                    uint left = 0,
                         right = g.offsets[u + 1] - g.offsets[u] - 1;
                    uint* u_nbr = g.neighbors + g.offsets[u];
                    int ux_id = check_exist(u_nbr, left, right, x);
                    // if x in the neighbor of u, then there is a butterfly
                    if (ux_id == -1) continue;
                    uint ux = g.edge_ids[g.offsets[u] + ux_id];
                    if (assign_edge[ux]) continue;
                    if (g.edge_support[ux] > g.edge_support[e_id])
                        g.edge_support[ux]--;
                    if (g.edge_support[wx] > g.edge_support[e_id])
                        g.edge_support[wx]--;
                    if (g.edge_support[wv] > g.edge_support[e_id])
                        g.edge_support[wv]--;
                }
            }

            assign_edge[e_id] = 1;
            assigned++;

            min_val = UINT32_MAX;
            for (auto e = 0; e < g.m; e++) {
                if (assign_edge[e]) continue;
                if (g.edge_support[e] < min_val) min_val = g.edge_support[e];
            }
        }
    }

    auto kmax =
            *std::max_element(g.bitruss, g.bitruss + g.m);
    log_info("k-bitruss max value %d", kmax);

#ifdef DISPLAY_RESULT
    for (auto e = 0; e < g.m; e++) {
        std::cout << "e " << e << " bitruss " << g.bitruss[e] << std::endl;
    }
#endif
}

// function for bitruss hindex decomposition
auto compute_bitruss_hindex(Graph& g, std::vector<uint>& prev,
                            std::vector<uint>& curr, const uint& e_id,
                            bool& converge) -> void {
    if (prev[e_id] == 0) return;

    auto ne = std::vector<uint>();

    uint u = g.edges[e_id * 2];
    uint v = g.edges[e_id * 2 + 1];

    uint const* u_nbr = g.neighbors + g.offsets[u];
    uint const u_nbr_len = g.offsets[u + 1] - g.offsets[u];

    uint const* v_nbr = g.neighbors + g.offsets[v];
    uint const v_nbr_len = g.offsets[v + 1] - g.offsets[v];

    for (uint i = 0; i < v_nbr_len; i++) {
        uint w = v_nbr[i];
        if (w == u) continue;
        uint wv = g.edge_ids[g.offsets[v] + i];


        uint* w_nbr = g.neighbors + g.offsets[w];
        uint w_nbr_len = g.offsets[w + 1] - g.offsets[w];

        uint j = 0;
        uint k = 0;

        while (j < w_nbr_len && k < u_nbr_len) {
            if (w_nbr[j] < u_nbr[k]) {
                j++;
            } else if (w_nbr[j] > u_nbr[k]) {
                k++;
            } else if (w_nbr[j] == u_nbr[k] && w_nbr[j] == v) {
                // get the edge id
                j++;
                k++;
            } else {
                // get edge id
                uint const wx = g.edge_ids[g.offsets[w] + j];
                uint const ux = g.edge_ids[g.offsets[u] + k];
                j++;
                k++;

                if (prev[ux] == 0 || prev[wx] == 0 || prev[wv] == 0) continue;
                auto min_val = std::min({prev[ux], prev[wx], prev[wv]});
                ne.push_back(min_val);
            }
        }
    }

    std::sort(ne.begin(), ne.end(), std::greater<>());

    uint h_index = 0;
    for (uint i = 0; i < ne.size(); i++) {
        if (ne[i] >= i + 1) {
            h_index = i + 1;
        } else {
            break;
        }
    }

#pragma omp critical
    curr[e_id] = h_index;

    if (h_index != prev[e_id]) {
#pragma omp critical
        converge = false;
    }
}

/**
 * bitruss decomposition based on h-index
 * @param g graph object
 */
auto c_bitruss_hindex(Graph& g, int threads) -> void {

    ebfc(&g, threads);
    log_info("running bitruss hindex decomposition with %d threads", threads);

    auto converge = false;
    auto prev = std::vector<uint>(g.m);
    std::copy(g.edge_support, g.edge_support + g.m, prev.begin());
    auto curr = std::vector<uint>(g.m, 0);

    omp_set_num_threads(threads);


    while (!converge) {
#pragma omp parallel
        {
#pragma atomic write
            converge = true;

#pragma omp for schedule(dynamic)
            for (uint uv = 0; uv < g.m; uv += 1) {
                compute_bitruss_hindex(g, prev, curr, uv, converge);
            }
#pragma omp barrier
#pragma omp for schedule(dynamic)
            for (auto e_id = 0; e_id < g.m; e_id++) {
                prev[e_id] = curr[e_id];
            }
        }

        curr = std::vector<uint>(g.m, 0);
    }

    std::copy(prev.begin(), prev.end(), g.bitruss);

    auto kmax =
            *std::max_element(g.bitruss, g.bitruss + g.m);
    log_info("k-bitruss max value %d", kmax);

}

auto process_support_zero(Graph&g, std::vector<bool>& processed) -> uint {
    uint count = 0;
#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (uint e = 0; e < g.m; e++) {
            if (g.edge_support[e] == 0) {
                processed[e] = true;
#pragma omp atomic
                count++;
            }
        }
    }

    return count;
}

auto scan_edge(Graph& g, std::vector<uint>& curr, std::vector<bool>& is_in_curr, std::vector<bool>& processed, int level) -> void {

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (uint e = 0; e < g.m; e++) {
            if (processed[e]) continue;
            if (g.edge_support[e] == level) {
                is_in_curr[e] = true;
#pragma omp critical
                curr.push_back(e);
            }
        }
    }
}


auto update_edge_support(uint eidx, int level, Graph& g, std::vector<uint>& next, std::vector<bool>& is_in_next) -> void {

    int curr;
#pragma omp atomic capture
    {
        curr = g.edge_support[eidx];
        g.edge_support[eidx]--;
    }


    if (curr == (level + 1)) {
#pragma omp critical
        is_in_next[eidx] = true;

#pragma omp critical
        next.push_back(eidx);
    }

    if (curr <= level) {
#pragma omp atomic
        g.edge_support[eidx]++;
    }
}


// function for peeling one edge
auto ppeel_one_edge(uint uv, uint wv, uint wx, uint ux, int level,
                    uint const is_peel_wv, uint const is_peel_wx, uint const is_peel_ux,
                    Graph& g, std::vector<uint>& next, std::vector<bool>& is_in_next) -> void {

    if (is_peel_wx) {
        if (uv < ux && ux < wv) update_edge_support(wx, level, g, next, is_in_next);
    }

    if (is_peel_wv) {
        if (uv < ux && ux < wx) update_edge_support(wv, level, g, next, is_in_next);
    }

    if (is_peel_ux) {
        if (uv < wv && wv < wx) update_edge_support(ux, level, g, next, is_in_next);
    }
}

// function for peeling two edge
auto ppeel_two_edge(uint uv, uint wv, uint wx, uint ux, int level,
                    uint const is_peel_wv, uint const is_peel_wx, uint const is_peel_ux,
                    Graph& g, std::vector<uint>& next, std::vector<bool>& is_in_next) -> void {
    // only peel one edge
    if (is_peel_wv && is_peel_wx) {
        if (uv < ux) {
            update_edge_support(wv, level, g, next, is_in_next);
            update_edge_support(wx, level, g, next, is_in_next);
        }
    }

    if (is_peel_wv && is_peel_ux) {
        if (uv < wx) {
            update_edge_support(wv, level, g, next, is_in_next);
            update_edge_support(ux, level, g, next, is_in_next);
        }
    }

    if (is_peel_wx && is_peel_ux) {
        if (uv < wv) {
            update_edge_support(wx, level, g, next, is_in_next);
            update_edge_support(ux, level, g, next, is_in_next);
        }
    }
}

// function for peeling three edge
auto ppeel_three_edge(uint wv, uint wx, uint ux, int level, Graph& g, std::vector<uint>& next, std::vector<bool>& is_in_next) -> void {
    // peel three edge
    update_edge_support(wv, level, g, next, is_in_next);
    update_edge_support(wx, level, g, next, is_in_next);
    update_edge_support(ux, level, g, next, is_in_next);
}

auto peel_edges(Graph& g, std::vector<uint>& curr, std::vector<bool>& is_in_curr, std::vector<bool>& processed,
                std::vector<uint>& next, std::vector<bool>& is_in_next, int level) -> void {

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (unsigned int uv : curr) {
            if (processed[uv]) continue;

            uint const u = g.edges[uv * 2];
            uint const v = g.edges[uv * 2 + 1];

            uint const* u_nbr = g.neighbors + g.offsets[u];
            uint const u_nbr_len = g.degrees[u];
            uint const* v_nbr = g.neighbors + g.offsets[v];

            for (uint x = 0; x < g.degrees[v]; x ++) {
                uint const w = v_nbr[x];
                if (w == u) continue;

                uint const wv = g.edge_ids[g.offsets[v] + x];
                if (processed[wv]) continue;

                uint const* w_nbr = g.neighbors + g.offsets[w];
                uint const w_nbr_len = g.degrees[w];

                uint j = 0;
                uint k = 0;

                while (j < w_nbr_len && k < u_nbr_len) {
                    if (w_nbr[j] < u_nbr[k]) {
                        j++;
                    } else if (w_nbr[j] > u_nbr[k]) {
                        k++;
                    } else if (w_nbr[j] == u_nbr[k] && w_nbr[j] == v) {
                        j++;
                        k++;
                    } else {
                        uint const wx = g.edge_ids[g.offsets[w] + j];
                        uint const ux = g.edge_ids[g.offsets[u] + k];
                        j++;
                        k++;

                        // peel edges
                        if (processed[ux] || processed[wx]) continue;

                        uint const is_peel_wv = !is_in_curr[wv];
                        uint const is_peel_wx = !is_in_curr[wx];
                        uint const is_peel_ux = !is_in_curr[ux];

                        uint const update_count = is_peel_wv + is_peel_wx + is_peel_ux;

                        if (update_count == 1) {
                            ppeel_one_edge(uv, wv, wx, ux, level, is_peel_wv, is_peel_wx, is_peel_ux, g, next, is_in_next);
                        } else if (update_count == 2) {
                            ppeel_two_edge(uv, wv, wx, ux, level, is_peel_wv, is_peel_wx, is_peel_ux, g, next, is_in_next);
                        } else if (update_count == 3) {
                            ppeel_three_edge(wv, wx, ux, level, g, next, is_in_next);
                        }
                    }
                }
            }
        }
    }
#pragma omp barrier

    // mark processed edges
    for (unsigned int uv : curr) {
        processed[uv] = true;
        is_in_curr[uv] = false;
    }
}

auto ppeel_edges(Graph& g, std::vector<uint>& curr, std::vector<bool>& is_in_curr, std::vector<bool>& processed,
                std::vector<uint>& next, std::vector<bool>& is_in_next, int level) -> void {

#pragma omp parallel
    {
        for (unsigned int uv : curr) {
            if (processed[uv]) continue;

            uint const u = g.edges[uv * 2];
            uint const v = g.edges[uv * 2 + 1];

            uint const* u_nbr = g.neighbors + g.offsets[u];
            uint const u_nbr_len = g.degrees[u];
            uint const* v_nbr = g.neighbors + g.offsets[v];

#pragma omp for schedule(static)
            for (uint x = 0; x < g.degrees[v]; x ++) {
                uint const w = v_nbr[x];
                if (w == u) continue;

                uint const wv = g.edge_ids[g.offsets[v] + x];
                if (processed[wv]) continue;

                uint const* w_nbr = g.neighbors + g.offsets[w];
                uint const w_nbr_len = g.degrees[w];

                uint j = 0;
                uint k = 0;

                while (j < w_nbr_len && k < u_nbr_len) {
                    if (w_nbr[j] < u_nbr[k]) {
                        j++;
                    } else if (w_nbr[j] > u_nbr[k]) {
                        k++;
                    } else if (w_nbr[j] == u_nbr[k] && w_nbr[j] == v) {
                        j++;
                        k++;
                    } else {
                        uint const wx = g.edge_ids[g.offsets[w] + j];
                        uint const ux = g.edge_ids[g.offsets[u] + k];
                        j++;
                        k++;

                        // peel edges
                        if (processed[ux] || processed[wx]) continue;

                        uint const is_peel_wv = !is_in_curr[wv];
                        uint const is_peel_wx = !is_in_curr[wx];
                        uint const is_peel_ux = !is_in_curr[ux];

                        uint const update_count = is_peel_wv + is_peel_wx + is_peel_ux;

                        if (update_count == 1) {
                            ppeel_one_edge(uv, wv, wx, ux, level, is_peel_wv, is_peel_wx, is_peel_ux, g, next, is_in_next);
                        } else if (update_count == 2) {
                            ppeel_two_edge(uv, wv, wx, ux, level, is_peel_wv, is_peel_wx, is_peel_ux, g, next, is_in_next);
                        } else if (update_count == 3) {
                            ppeel_three_edge(wv, wx, ux, level, g, next, is_in_next);
                        }
                    }
                }
            }
        }
    }
#pragma omp barrier

    // mark processed edges
    for (unsigned int uv : curr) {
        processed[uv] = true;
        is_in_curr[uv] = false;
    }
}


auto ccompress_graph(Graph *g, std::vector<uint> &neighbors, std::vector<uint> &degrees, std::vector<uint> &edges_ids, int level) -> uint {
    uint m = 0;
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (auto u = 0; u < g->n; u++) {
            uint *u_nbr = g->neighbors + g->offsets[u];
            uint u_nbr_len = g->offsets[u + 1] - g->offsets[u];

            uint idx = 0;
            for (auto i = 0; i < u_nbr_len; i++) {
                uint v = u_nbr[i];
                uint uv = g->edge_ids[g->offsets[u] + i];

                // check if the edge is in the current edge array that need to be peeled
                if (g->edge_support[uv] > level) {
                    neighbors[g->offsets[u] + idx] = v;
                    edges_ids[g->offsets[u] + idx] = uv;
                    idx++;
                }
            }
            degrees[u] = idx;
#pragma omp atomic
            m += idx;
        }
    };

#pragma barrier
    // return the number of edges
    return m / 2;
}

/**
 * bitruss decomposition peeling-based on cpu
 * @param g graph object
 */
auto c_bitruss_msp(Graph* g, int threads) -> void {
    // run butterfly counting for each edge
    ebfc(g, threads);

    log_info("running bitruss msp decomposition with %d threads", threads);

    auto curr = std::vector<uint>();
    auto next = std::vector<uint>();
    auto is_in_curr = std::vector<bool>(g->m, false);
    auto is_in_next = std::vector<bool>(g->m, false);
    auto processed = std::vector<bool>(g->m, false);

    int level = 1;
    uint m = g->m;
    auto neighbors = std::vector<uint>(g->m * 2);
    auto degree = std::vector<uint>(g->n);
    auto edges_id = std::vector<uint>(g->m * 2);


    int edge_num = int(g->m);
    edge_num -= int(process_support_zero(*g, processed));

    while (edge_num > 0) {
        scan_edge(*g, curr, is_in_curr, processed, level);

        while (!curr.empty()) {
            edge_num -= int(curr.size());
            // then peeling edges
            if (curr.size() > threads * 3) {
                peel_edges(*g, curr, is_in_curr, processed, next, is_in_next, level);
            } else {
                ppeel_edges(*g, curr, is_in_curr, processed, next, is_in_next, level);
            }

#pragma omp barrier

            std::swap(curr, next);
            std::swap(is_in_curr, is_in_next);

            next.resize(0);
            is_in_next = std::vector<bool>(g->m, false);
        }

        if (edge_num <= m * 0.5 && m >= g->m * 0.1) {
            m = ccompress_graph(g, neighbors, degree, edges_id, level);
            std::copy(neighbors.begin(), neighbors.end(), g->neighbors);
            std::copy(degree.begin(), degree.end(), g->degrees);
            std::copy(edges_id.begin(), edges_id.end(), g->edge_ids);

            degree.clear();
            neighbors.clear();
            edges_id.clear();

        }
        level += 1;
    }


    log_info("bitruss msp with %'d levels", level - 1);


}
