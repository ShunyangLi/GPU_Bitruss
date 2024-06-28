#include "bitruss.cuh"

/**
* binary search find if x in the neighbor of u,
* if x in the neighbor of u, then return the neighbor position of x in the neighbor of u
* if x not in the neighbor of u, then return -1
* @param nbr neighbor array
* @param left left index
* @param right right index
* @param value the value to find
* @return idx or -1
*/
__inline__ __device__ auto binary_search(const uint *nbr, uint left, uint right, uint value) -> int {
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

// count edge support with 0
__global__ auto count_es_zero(uint num_edge, const int *d_edge_support, bool *processed, int *d_count) -> void {
    uint e_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    int local_count = 0;
    for (uint i = e_id; i < num_edge; i += stride) {
        if (d_edge_support[i] == 0 && !processed[i]) {
            local_count++;
            processed[i] = true;
        }
    }
    atomicAdd(d_count, local_count);
}

auto process_es_zero(uint num_edge, const int *d_edge_support, bool *processed) -> int {
    dim3 blockSize(64);
    dim3 gridSize((num_edge + blockSize.x - 1) / blockSize.x);
    int count;
    int *d_count;

    cudaMalloc((void **) &d_count, sizeof(int));
    count_es_zero<<<gridSize, blockSize>>>(num_edge, d_edge_support, processed, d_count);
    cudaDeviceSynchronize();
    cudaMemcpy((void *) &count, d_count, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    return count;
}

/**
* scan the edges with support <= k, cuda function
* @param num_edge the number of edges
* @param level the level of support
* @param d_edge_support the edge support array
* @param curr the current edge array, need to peeled
* @param curr_idx  the index of current edge array
* @param is_in_curr if the edge is in the current edge array
* @param processed edges already processed
*/
__global__ auto scan_sub_kernel(uint num_edge, uint level, const int *d_edge_support,
                                uint *curr, int *curr_idx, bool *is_in_curr,
                                const bool *processed) -> void {

    uint e_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for (auto i = e_id; i < num_edge; i += stride) {
        if (i >= num_edge) continue;

        if (d_edge_support[i] == level && !processed[i]) {
            uint const curr_idx_val = atomicAdd(curr_idx, 1);
            curr[curr_idx_val] = i;
            is_in_curr[i] = true;
        }
    }
}

// scan all the edges with support = level
auto scan_sub_level(uint num_edge, uint level, const int *d_edge_support,
                    uint *curr, int *curr_idx, bool *is_in_curr,
                    bool *processed) -> void {

    dim3 blockSize(BLK_DIM);
    dim3 gridSize((num_edge + blockSize.x - 1) / blockSize.x);

    scan_sub_kernel<<<gridSize, blockSize>>>(num_edge, level, d_edge_support,
                                             curr, curr_idx, is_in_curr, processed);
    cudaDeviceSynchronize();
}


// update edge support
__inline__ __device__ auto update_edge_support(uint eidx, int level, int *d_edge_support,
                                               uint *next, int *next_idx, bool *is_in_next) -> void {

    int cur = atomicSub(&d_edge_support[eidx], 1);
    if (cur == (level + 1)) {
        uint insert_inx = atomicAdd(next_idx, 1);
        next[insert_inx] = eidx;
        is_in_next[eidx] = true;
    }

    if (cur <= level) {
        atomicAdd(&d_edge_support[eidx], 1);
    }
}

// function for peeling one edge
__inline __device__ auto peel_one_edge(uint uv, uint wv, uint wx, uint ux, int level, int *d_edge_support,
                                       uint *next, int *next_idx, bool *is_in_next, const bool *is_in_curr,
                                       uint const is_peel_wv, uint const is_peel_wx, uint const is_peel_ux) -> void {
    //    auto is_peel_wv = !is_in_curr[wv];
    //    auto is_peel_wx = !is_in_curr[wx];
    //    auto is_peel_ux = !is_in_curr[ux];
    // only peel one edge

    if (is_peel_wx) {
        if (uv < ux && ux < wv) update_edge_support(wx, level, d_edge_support, next, next_idx, is_in_next);
    }

    if (is_peel_wv) {
        if (uv < ux && ux < wx) update_edge_support(wv, level, d_edge_support, next, next_idx, is_in_next);
    }

    if (is_peel_ux) {
        if (uv < wv && wv < wx) update_edge_support(ux, level, d_edge_support, next, next_idx, is_in_next);
    }
}

// function for peeling two edge
__inline __device__ auto peel_two_edge(uint uv, uint wv, uint wx, uint ux, int level, int *d_edge_support,
                                       uint *next, int *next_idx, bool *is_in_next, const bool *is_in_curr,
                                       uint const is_peel_wv, uint const is_peel_wx, uint const is_peel_ux) -> void {
    //    auto is_peel_wv = !is_in_curr[wv];
    //    auto is_peel_wx = !is_in_curr[wx];
    //    auto is_peel_ux = !is_in_curr[ux];
    // only peel one edge
    if (is_peel_wv && is_peel_wx) {
        if (uv < ux) {
            update_edge_support(wv, level, d_edge_support, next, next_idx, is_in_next);
            update_edge_support(wx, level, d_edge_support, next, next_idx, is_in_next);
        }
    }

    if (is_peel_wv && is_peel_ux) {
        if (uv < wx) {
            update_edge_support(wv, level, d_edge_support, next, next_idx, is_in_next);
            update_edge_support(ux, level, d_edge_support, next, next_idx, is_in_next);
        }
    }

    if (is_peel_wx && is_peel_ux) {
        if (uv < wv) {
            update_edge_support(wx, level, d_edge_support, next, next_idx, is_in_next);
            update_edge_support(ux, level, d_edge_support, next, next_idx, is_in_next);
        }
    }
}

// function for peeling three edge
__inline __device__ auto peel_three_edge(uint wv, uint wx, uint ux, int level, int *d_edge_support,
                                         uint *next, int *next_idx, bool *is_in_next) -> void {
    // peel three edge
    update_edge_support(wv, level, d_edge_support, next, next_idx, is_in_next);
    update_edge_support(wx, level, d_edge_support, next, next_idx, is_in_next);
    update_edge_support(ux, level, d_edge_support, next, next_idx, is_in_next);
}


__global__ auto process_sub_level_kernel(int level, const uint *d_edges,
                                         const uint *d_edge_ids, const uint *d_offset,
                                         uint *d_neighbors, int *d_edge_support,
                                         const uint *curr, const int *curr_idx, const bool *is_in_curr,
                                         uint *next, int *next_idx, bool *is_in_next,
                                         const bool *processed, const uint *d_degree) -> void {
    uint const d_idx_curr = blockIdx.x;
    uint const stride = gridDim.x;
    uint const dim = blockDim.x;
    uint const tid = threadIdx.x;
    //    __shared__ uint cache[SHARE_MEM];
    __shared__ uint *u_nbr;
    __shared__ uint *v_nbr;

    for (uint idx_curr = d_idx_curr; idx_curr < *curr_idx; idx_curr += stride) {
        if (idx_curr >= *curr_idx) return;

        uint const uv = curr[idx_curr];
        if (processed[uv]) return;

        uint const u = d_edges[uv * 2];
        uint const v = d_edges[uv * 2 + 1];

        uint v_nbr_len = d_degree[v];
        uint u_nbr_len = d_degree[u];

        if (threadIdx.x == 0) {
            u_nbr = d_neighbors + d_offset[u];
            v_nbr = d_neighbors + d_offset[v];
        }

        __syncthreads();

        for (uint i = tid; i < v_nbr_len; i += dim) {
            //            uint v_id = i;
            if (i >= v_nbr_len) break;

            uint const w = v_nbr[i];
            if (w == u) continue;

            uint const wv = d_edge_ids[d_offset[v] + i];
            if (processed[wv]) continue;
            if (is_in_curr[wv] && uv > wv) continue;


            uint const *w_nbr = d_neighbors + d_offset[w];
            uint const w_nbr_len = d_degree[w];

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
                    uint const wx = d_edge_ids[d_offset[w] + j];
                    uint const ux = d_edge_ids[d_offset[u] + k];
                    j++;
                    k++;

                    if (processed[wx] || processed[ux]) continue;
                    // check if the edge is in the current edge array that need to be peeled
                    uint const is_peel_wv = !is_in_curr[wv];
                    uint const is_peel_wx = !is_in_curr[wx];
                    uint const is_peel_ux = !is_in_curr[ux];

                    uint const update_count = is_peel_wv + is_peel_wx + is_peel_ux;

                    if (update_count == 1) {
                        peel_one_edge(uv, wv, wx, ux, level, d_edge_support, next, next_idx, is_in_next, is_in_curr, is_peel_wv, is_peel_wx, is_peel_ux);
                    } else if (update_count == 2) {
                        peel_two_edge(uv, wv, wx, ux, level, d_edge_support, next, next_idx, is_in_next, is_in_curr, is_peel_wv, is_peel_wx, is_peel_ux);
                    } else if (update_count == 3) {
                        peel_three_edge(wv, wx, ux, level, d_edge_support, next, next_idx, is_in_next);
                    }
                }
            }

        }
    }
}


__global__ auto update_processed_edges(bool *dev_in_curr, const int *curr_idx, const uint *dev_curr, bool *dev_processed) -> void {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for (uint i = idx; i < *curr_idx; i += stride) {
        uint e_id = dev_curr[i];
        dev_processed[e_id] = true;
        dev_in_curr[e_id] = false;
    }
}


// process all the edges with support = level
auto process_sub_level(int level, const uint *d_edges,
                       const uint *d_edge_ids, const uint *d_offset,
                       uint *d_neighbors, int *d_edge_support,
                       uint *curr, int *curr_idx, bool *is_in_curr,
                       uint *next, int *next_idx, bool *is_in_next,
                       bool *processed, uint nums_in_curr, const uint *d_degree) -> void {

    dim3 blockSize(BLK_DIM);
    dim3 gridSize(nums_in_curr);

    process_sub_level_kernel<<<gridSize, blockSize>>>(level, d_edges,
                                                      d_edge_ids, d_offset, d_neighbors, d_edge_support,
                                                      curr, curr_idx, is_in_curr, next, next_idx, is_in_next, processed, d_degree);

    cudaDeviceSynchronize();

    update_processed_edges<<<gridSize, blockSize>>>(is_in_curr, curr_idx, curr, processed);

    cudaDeviceSynchronize();
}

auto compress_graph(Graph *g, std::vector<uint> &neighbors, std::vector<uint> &degrees, std::vector<uint> &edges_ids, int level) -> uint {
    uint m = 0;
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
        m += idx;
    }
    // return the number of edges
    return m / 2;
}

// kernel for compress the graph
__global__ auto graph_compress_kernel(const uint *d_degree, uint *dt_degree, uint *d_neighbors,
                                      uint *dt_neighbors, const uint *d_edge_ids, uint *dt_edge_ids,
                                      uint *d_m, const int *d_edge_support, uint *d_offset,
                                      uint const num_vertex, int const level) -> void {

    uint u = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for (uint i = u; i < num_vertex; i += stride) {
        uint *u_nbr = d_neighbors + d_offset[i];

        uint idx = 0;
        for (uint j = 0; j < d_degree[i]; j++) {
            uint v = u_nbr[j];
            uint uv = d_edge_ids[d_offset[i] + j];

            if (d_edge_support[uv] > level) {
                dt_neighbors[d_offset[i] + idx] = v;
                dt_edge_ids[d_offset[i] + idx] = uv;
                idx++;
            }
        }

        dt_degree[i] = idx;
        atomicAdd(d_m, idx);
    }
}


__global__ auto process_edge(uint m, uint* d_edges, uint const* d_degree) -> void {
    uint eid = blockIdx.x * blockDim.x + threadIdx.x;

    if (eid >= m) return;
    uint u = d_edges[eid * 2];
    uint v = d_edges[eid * 2 + 1];

    if (d_degree[u] > d_degree[v] * 10) {
        uint tmp = u;
        d_edges[eid * 2] = v;
        d_edges[eid * 2 + 1] = tmp;
    }
}

// pre-processing the edges
__global__ auto nprocess_edge(uint m, uint* d_edges, uint const* d_degree, const uint *d_neighbors, const uint *d_offset) -> void {
    uint const d_idx_curr = blockIdx.x;
    uint const dim = blockDim.x;
    uint const tid = threadIdx.x;
    __shared__ ull u_val;
    __shared__ ull v_val;

    if (d_idx_curr >= m) return;

    if (tid == 0) {
        u_val = 0;
        v_val = 0;
    }

    __syncthreads();

    uint const eid = d_idx_curr;
    uint const u = d_edges[eid * 2];
    uint const v = d_edges[eid * 2 + 1];

    uint const* u_nbr = d_neighbors + d_offset[u];
    uint const* v_nbr = d_neighbors + d_offset[v];

    if (d_degree[u] < 10 || d_degree[v] < 10) return;

    for (uint i = tid; i < d_degree[u]; i += dim) {
        uint const w = u_nbr[i];
        atomicAdd(&u_val, d_degree[w] + d_degree[v]);
    }

    for (uint i = tid; i < d_degree[v]; i += dim) {
        uint const w = v_nbr[i];
        atomicAdd(&v_val, d_degree[w] + d_degree[u]);
    }

    __syncthreads();

    if (tid == 0) {
        if (u_val > v_val * 100) {
            uint tmp = u;
            d_edges[eid * 2] = v;
            d_edges[eid * 2 + 1] = tmp;
        }
    }

}

/**
* muitlple stage bitruss decomposition online peeling
* While
*      1. scan the edges with support <= k
*      2. process edges and update support
*      3. update k ++
* @param g
*/
auto bitruss_msp(Graph *g) -> void {

    //    ebfc(g, 48);
    // TODO: add butterfly cache
    bfc_evpp(g);
    log_info("running bitruss decomposition online peeling");

    auto node_num = g->u_num + g->l_num;
    int *d_edge_support;
    uint *d_edges, *d_offset, *d_neighbors, *d_edge_ids;
    uint *d_degree;

    // for graph data
    cudaMalloc((void **) &d_edges, sizeof(uint) * g->m * 2);
    cudaMalloc((void **) &d_offset, sizeof(uint) * (node_num + 1));
    cudaMalloc((void **) &d_neighbors, sizeof(uint) * g->m * 2);
    cudaMalloc((void **) &d_edge_ids, sizeof(uint) * g->m * 2);
    cudaMalloc((void **) &d_edge_support, sizeof(int) * g->m);
    cudaMalloc((void **) &d_degree, sizeof(uint) * g->n);

    cudaMemcpy((void *) d_edges, (void *) g->edges, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_offset, (void *) g->offsets, sizeof(uint) * (node_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_neighbors, (void *) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_edge_ids, (void *) g->edge_ids, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_edge_support, (void *) g->edge_support, sizeof(int) * g->m, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_degree, (void *) g->degrees, sizeof(uint) * g->n, cudaMemcpyHostToDevice);

    // used data in algorithm
    uint *d_curr, *d_next;
    int *d_curr_idx, *d_next_idx;
    bool *is_in_curr, *is_in_next, *d_processed;
    // allocate memory
    cudaMalloc((void **) &d_processed, sizeof(bool) * g->m);
    cudaMalloc((void **) &d_curr, sizeof(uint) * g->m);
    cudaMalloc((void **) &d_next, sizeof(uint) * g->m);
    cudaMalloc((void **) &d_curr_idx, sizeof(int));
    cudaMalloc((void **) &d_next_idx, sizeof(int));
    cudaMalloc((void **) &is_in_curr, sizeof(bool) * g->m);
    cudaMalloc((void **) &is_in_next, sizeof(bool) * g->m);

    // init memory
    cudaMemset(d_processed, 0, sizeof(bool) * g->m);
    cudaMemset(d_curr_idx, 0, sizeof(int));
    cudaMemset(d_next_idx, 0, sizeof(int));
    cudaMemset(is_in_curr, false, sizeof(bool) * g->m);
    cudaMemset(is_in_next, false, sizeof(bool) * g->m);

    /**
    * using cache requires the following variables:
    * is_in_cache: the edges in the cache
    * d_cache: the edges in the cache
    * d_cache_idx: the index of the cache
    * d_cache_edges: the number of edges in the cache
    */

    // new edges support bucket
    auto neighbors = std::vector<uint>(g->m * 2);
    auto degree = std::vector<uint>(g->n);
    auto edges_id = std::vector<uint>(g->m * 2);
    uint m = g->m;

    int edge_num = int(g->m);
    int level = 1;
    int curr_idx = 0;
    // remove all edges with support 0
    int num_es_zero = process_es_zero(g->m, d_edge_support, d_processed);
    edge_num -= num_es_zero;

//    nprocess_edge<<<g->m, BLK_DIM>>>(g->m, d_edges, d_degree, d_neighbors, d_offset);
    cudaDeviceSynchronize();

    while (edge_num > 0) {
        scan_sub_level(g->m, level, d_edge_support, d_curr, d_curr_idx, is_in_curr, d_processed);
        // get the number of edges with support <= level
        cudaMemcpy((void *) &curr_idx, (void *) d_curr_idx, sizeof(int), cudaMemcpyDeviceToHost);

        while (curr_idx > 0) {
            if (edge_num == curr_idx) {
                edge_num = 0;
                break;
            }

            edge_num -= curr_idx;
            process_sub_level(level, d_edges, d_edge_ids, d_offset, d_neighbors, d_edge_support,
                              d_curr, d_curr_idx, is_in_curr, d_next, d_next_idx, is_in_next, d_processed, curr_idx, d_degree);

            // then add the sub_level_process function
            cudaMemcpy((void *) d_curr, (void *) d_next, sizeof(uint) * g->m, cudaMemcpyDeviceToDevice);
            cudaMemcpy((void *) is_in_curr, (void *) is_in_next, sizeof(bool) * g->m, cudaMemcpyDeviceToDevice);
            cudaMemcpy((void *) d_curr_idx, (void *) d_next_idx, sizeof(uint), cudaMemcpyDeviceToDevice);

            cudaMemset((void *) is_in_next, false, sizeof(bool) * g->m);
            cudaMemset((void *) d_next_idx, 0, sizeof(uint));
            cudaMemcpy((void *) &curr_idx, (void *) d_curr_idx, sizeof(int), cudaMemcpyDeviceToHost);
        }

        // compress the graph
        if (edge_num <= m * 0.5 && m >= g->m * 0.1) {
            cudaMemcpy((void *) g->edge_support, (void *) d_edge_support, sizeof(int) * g->m, cudaMemcpyDeviceToHost);
            m = compress_graph(g, neighbors, degree, edges_id, level);
            // copy neighbors to d_neighbors
            cudaMemcpy((void *) d_neighbors, (void *) neighbors.data(), sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
            cudaMemcpy((void *) d_edge_ids, (void *) edges_id.data(), sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
            // copy degree to d_degree
            cudaMemcpy((void *) d_degree, (void *) degree.data(), sizeof(uint) * g->n, cudaMemcpyHostToDevice);

            degree.clear();
            neighbors.clear();
            edges_id.clear();
        }

        level += 1;
    }

    log_info("bitruss msp with %'d levels", level - 1);

    // copy d_edge_support to host
    memset(g->bitruss, 0, sizeof(uint) * g->m);
    cudaMemcpy((void *) g->bitruss, (void *) d_edge_support, sizeof(uint) * g->m, cudaMemcpyDeviceToHost);

    // check if there is any error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_trace("CUDA error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // free memory
    cudaFree(d_edges);
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_edge_ids);
    cudaFree(d_edge_support);

    cudaFree(d_processed);
    cudaFree(d_curr);
    cudaFree(d_next);
    cudaFree(d_curr_idx);
    cudaFree(d_next_idx);
    cudaFree(is_in_curr);
    cudaFree(is_in_next);
    cudaFree(d_degree);
}


/**
* muitlple stage bitruss decomposition online peeling
* While
*      1. scan the edges with support <= k
*      2. process edges and update support
*      3. update k ++
* @param g
*/
auto bitruss_mspp(Graph *g) -> void {

    bfc_evpp(g);
    log_info("running bitruss decomposition online peeling (baseline)");



    auto node_num = g->u_num + g->l_num;
    int *d_edge_support;
    uint *d_edges, *d_offset, *d_neighbors, *d_edge_ids;
    uint *d_degree;

    // for graph data
    cudaMalloc((void **) &d_edges, sizeof(uint) * g->m * 2);
    cudaMalloc((void **) &d_offset, sizeof(uint) * (node_num + 1));
    cudaMalloc((void **) &d_neighbors, sizeof(uint) * g->m * 2);
    cudaMalloc((void **) &d_edge_ids, sizeof(uint) * g->m * 2);
    cudaMalloc((void **) &d_edge_support, sizeof(int) * g->m);
    cudaMalloc((void **) &d_degree, sizeof(uint) * g->n);

    cudaMemcpy((void *) d_edges, (void *) g->edges, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_offset, (void *) g->offsets, sizeof(uint) * (node_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_neighbors, (void *) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_edge_ids, (void *) g->edge_ids, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_edge_support, (void *) g->edge_support, sizeof(int) * g->m, cudaMemcpyHostToDevice);
    cudaMemcpy((void *) d_degree, (void *) g->degrees, sizeof(uint) * g->n, cudaMemcpyHostToDevice);

    // used data in algorithm
    uint *d_curr, *d_next;
    int *d_curr_idx, *d_next_idx;
    bool *is_in_curr, *is_in_next, *d_processed;
    // allocate memory
    cudaMalloc((void **) &d_processed, sizeof(bool) * g->m);
    cudaMalloc((void **) &d_curr, sizeof(uint) * g->m);
    cudaMalloc((void **) &d_next, sizeof(uint) * g->m);
    cudaMalloc((void **) &d_curr_idx, sizeof(int));
    cudaMalloc((void **) &d_next_idx, sizeof(int));
    cudaMalloc((void **) &is_in_curr, sizeof(bool) * g->m);
    cudaMalloc((void **) &is_in_next, sizeof(bool) * g->m);

    // init memory
    cudaMemset(d_processed, 0, sizeof(bool) * g->m);
    cudaMemset(d_curr_idx, 0, sizeof(int));
    cudaMemset(d_next_idx, 0, sizeof(int));
    cudaMemset(is_in_curr, false, sizeof(bool) * g->m);
    cudaMemset(is_in_next, false, sizeof(bool) * g->m);

    /**
    * using cache requires the following variables:
    * is_in_cache: the edges in the cache
    * d_cache: the edges in the cache
    * d_cache_idx: the index of the cache
    * d_cache_edges: the number of edges in the cache
    */

    // process the edges
    process_edge<<<(g->m + BLK_DIM - 1) / BLK_DIM, BLK_DIM>>>(g->m, d_edges, d_degree);

    int edge_num = int(g->m);
    int level = 1;
    int curr_idx = 0;
    // remove all edges with support 0
    int num_es_zero = process_es_zero(g->m, d_edge_support, d_processed);
    edge_num -= num_es_zero;

    while (edge_num > 0) {
        scan_sub_level(g->m, level, d_edge_support, d_curr, d_curr_idx, is_in_curr, d_processed);
        // get the number of edges with support <= level
        cudaMemcpy((void *) &curr_idx, (void *) d_curr_idx, sizeof(int), cudaMemcpyDeviceToHost);

        while (curr_idx > 0) {
            if (edge_num == curr_idx) {
                edge_num = 0;
                break;
            }

            edge_num -= curr_idx;
            process_sub_level(level, d_edges, d_edge_ids, d_offset, d_neighbors, d_edge_support,
                              d_curr, d_curr_idx, is_in_curr, d_next, d_next_idx, is_in_next, d_processed, curr_idx, d_degree);

            // then add the sub_level_process function
            cudaMemcpy((void *) d_curr, (void *) d_next, sizeof(uint) * g->m, cudaMemcpyDeviceToDevice);
            cudaMemcpy((void *) is_in_curr, (void *) is_in_next, sizeof(bool) * g->m, cudaMemcpyDeviceToDevice);
            cudaMemcpy((void *) d_curr_idx, (void *) d_next_idx, sizeof(uint), cudaMemcpyDeviceToDevice);

            cudaMemset((void *) is_in_next, false, sizeof(bool) * g->m);
            cudaMemset((void *) d_next_idx, 0, sizeof(uint));
            cudaMemcpy((void *) &curr_idx, (void *) d_curr_idx, sizeof(int), cudaMemcpyDeviceToHost);
        }

        level += 1;
    }

    log_info("(baseline) bitruss msp with %'d levels", level - 1);

    // copy d_edge_support to host
    memset(g->bitruss, 0, sizeof(uint) * g->m);
    cudaMemcpy((void *) g->bitruss, (void *) d_edge_support, sizeof(uint) * g->m, cudaMemcpyDeviceToHost);

    // check if there is any error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_trace("CUDA error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // free memory
    cudaFree(d_edges);
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_edge_ids);
    cudaFree(d_edge_support);

    cudaFree(d_processed);
    cudaFree(d_curr);
    cudaFree(d_next);
    cudaFree(d_curr_idx);
    cudaFree(d_next_idx);
    cudaFree(is_in_curr);
    cudaFree(is_in_next);
    cudaFree(d_degree);
}